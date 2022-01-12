// be.cpp: Back end entry point for BNHSA_Z.
// Copyright (C) 2021  erikoui

// Possible enhancement: render stuff in the terminal with this
// https://github.com/Victormeriqui/Consol3

// NOTES ON Diaphragm:
// Apply the lateral node force to all nodes and apply moment to the nodes according to the perpendicular distance, as well.

#include <iostream>
#include <algorithm>
#include <String>
#include <iomanip>
#include <fstream>
#include <vector>

#include <math.h>
#include <windows.h>
#include <stdlib.h>

// I dont know how to setup include directories
#include "C:/eigen/Eigen/Dense"
#include "C:/eigen/Eigen/Sparse"
#include "C:/eigen/Eigen/StdVector"

// Nlohmann json library
#include "json.hpp"


//Im using euler-bernoulli beams, but this might be useful if i decide to change to timoshenko (which i should)
#define shear_lock_coeff 5 / 6

#define NONE 0
#define ERRORS 1
#define INFO 2
#define VERBOSE 3

using namespace Eigen;
using json = nlohmann::json;

int g_log_level;

class BNHParameters {
public:
    int max_iter = -1; // -1 means dont do geometry optimization thing
    bool calcCOR = false;
    bool calcDisplacements = false;
    std::string original_fn;

    BNHParameters() {
        max_iter = -1;
        calcCOR = false;
        calcDisplacements = false;
        original_fn = "ERR_FN_NOT_SET";
    }
};

bool log(int level) {
    return level <= g_log_level;
}

void print2DVectorDouble(std::vector<std::vector<double> > N) {
    for (int i = 0;i < N.size();i++) {
        for (int j = 0;j < N[i].size();j++) {
            std::cout << std::setprecision(4) << N[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}

void print2DVectorInt(std::vector<std::vector<int> > N) {
    for (int i = 0;i < N.size();i++) {
        for (int j = 0;j < N[i].size();j++) {
            std::cout << N[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}

char* getCmdOption(char** begin, char** end, const std::string& option) {
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

void printAboutBox() {
    std::cout << std::endl
        << std::endl
        << "               BNHSA_Z Copyright (C) 2021 erikoui" << std::endl
        << "         This program comes with ABSOLUTELY NO WARRANTY;" << std::endl
        << "     This is free software, and you are welcome to redistribute it" << std::endl
        << "               under the conditions of GNU GPL 3.0." << std::endl
        << std::endl
        << "  Usage:" << std::endl
        << "  -f filename   | path to the json file" << std::endl
        << "  -v log_level  | log level: 0 for none, 1 for errors, 2 for info, 3 for verbose (default 2)" << std::endl
        << "  -c            | calculate center of rigidity and exit" << std::endl
        << "  -s            | solve with the boundary conditions and forces from the json file and exit" << std::endl
        << "  -o max_iter   | optimize geometry" << std::endl
        << std::endl;
}

BNHParameters parseCLIArgs(int argc, char* argv[]) {
    // Parses Command Line Arguments

    BNHParameters parameters;

    // Set default log level
    g_log_level = INFO;

    // Show about box
    if ((argc <= 1) || cmdOptionExists(argv, argv + argc, "-h")) {
        printAboutBox();
    }

    // Get input filename
    char* in_filename = getCmdOption(argv, argv + argc, "-f");
    if (!in_filename) {
        std::cout << "No file specified.";
        exit(1);
    } else {
        parameters.original_fn = std::string(in_filename);
        if (parameters.original_fn.substr(parameters.original_fn.size() - 5, parameters.original_fn.size()) != ".json") {
            std::cout << "Not a .json file.";
            exit(1);
        }
    }

    // Set log level
    char* log_level_cp = getCmdOption(argv, argv + argc, "-v");
    if (g_log_level) {
        g_log_level = atoi(log_level_cp);
        if (g_log_level < NONE || g_log_level > VERBOSE) {
            std::cout << "Invalid log level.";
            exit(1);
        }
    }

    int mutuallyExclusive = 0;

    // Only calculate center of rotation stuff
    parameters.calcCOR = cmdOptionExists(argv, argv + argc, "-c");
    if (parameters.calcCOR)
        mutuallyExclusive++;

    // Only calculate displacements once
    parameters.calcDisplacements = cmdOptionExists(argv, argv + argc, "-s");
    if (parameters.calcDisplacements)
        mutuallyExclusive++;

    char* max_iter_cp = getCmdOption(argv, argv + argc, "-o");
    if (max_iter_cp) {
        parameters.max_iter = atoi(max_iter_cp);
        mutuallyExclusive++;
    }

    if (mutuallyExclusive > 1) {
        std::cout << "More than one operation mode specified (-c, -s, -o).";
        exit(1);
    }

    return parameters;
}

json loadJSONFromFile(std::string in_filename) {
    // Read entire file into memory
    json modelDb;
    if (log(INFO))
        std::cout << "Loading file " << in_filename << std::endl;
    std::string jsonRaw;
    std::ifstream file(in_filename);
    if (file.is_open()) {
        // Allocate memory
        file.seekg(0, std::ios::end);
        jsonRaw.reserve(file.tellg());
        file.seekg(0, std::ios::beg);
        // Save the data in jsonRaw
        jsonRaw.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    } else {
        if (log(ERRORS))
            std::cout << "FATAL: Error opening file." << std::endl;
        exit(1);
    }

    // Parse jsonRaw to json object
    try {
        modelDb = json::parse(jsonRaw);
        jsonRaw.clear();
        // TODO: Check if all data is here after loading JSON
        if (modelDb["FEnodes"].size() == 0) {
            throw std::runtime_error("No nodes defined.");
        }
    }
    catch (const std::exception& e) {
        if (log(ERRORS))
            std::cout << "FATAL: Error in file data: " << e.what() << std::endl;
        exit(1);
    }

    if (log(INFO))
        std::cout << std::endl << "File loaded." << std::endl;

    return modelDb;
}

std::vector<std::vector<double> > loadNodeMatrix(json modelDb, int nNodes) {
    if (log(INFO)) {
        std::cout << std::endl << "Loading Node Matrix..." << std::endl;
    }
    std::vector<std::vector<double> > N(nNodes, std::vector<double>(3));
    for (int i = 0; i < N.size(); i++) {
        for (int j = 0; j < N[i].size(); j++) {
            N[i][j] = modelDb["FEnodes"][i]["coords"][j];
        }
    }
    if (log(VERBOSE)) {
        std::cout << std::endl << "Node matrix:" << std::endl;
        print2DVectorDouble(N);
    }
    return N;
}

std::vector<std::vector<int> > loadConnectivityMatrix(json modelDb, int nMembers) {
    if (log(INFO)) {
        std::cout << std::endl << "Loading Connectivity matrix..." << std::endl;
    }
    std::vector<std::vector<int> > C(nMembers, std::vector<int>(2));
    for (int i = 0; i < nMembers; i++) {
        C[i][0] = modelDb["FEmembers"][i]["from"];
        C[i][1] = modelDb["FEmembers"][i]["to"];
    }
    if (log(VERBOSE)) {
        std::cout << std::endl << "Connectivity matrix:" << std::endl;
        print2DVectorInt(C);
    }
    return C;
}

std::vector<MatrixXd> loadRotatedLocalStiffnessMatrices(json modelDb, int nMembers, std::vector<std::vector<double> > N, std::vector<std::vector<int> > C, int nNodes, std::vector<MatrixXd>& rotationMatrices) {
    if (log(INFO)) {
        std::cout << std::endl << "Calculating local stiffness matrices..." << std::endl;
    }
    std::vector<MatrixXd> K_rotated_local(nMembers, MatrixXd(12, 12));
    MatrixXd K_local;
    // Material properties
    std::vector<double> E(nMembers);
    std::vector<double> Ix(nMembers);
    std::vector<double> Iy(nMembers);
    std::vector<double> J(nMembers);
    std::vector<double> A(nMembers);
    std::vector<double> L(nMembers);
    std::vector<double> G(nMembers);
    std::vector<MatrixXd> Rx(nMembers, MatrixXd(3, 3));
    std::vector<MatrixXd> Ry(nMembers, MatrixXd(3, 3));
    std::vector<MatrixXd> Rz(nMembers, MatrixXd(3, 3));

    for (int i = 0; i < nMembers; i++) {
        E[i] = modelDb["FEmembers"][i]["E"];
        Ix[i] = modelDb["FEmembers"][i]["Ix"];
        Iy[i] = modelDb["FEmembers"][i]["Iy"];
        J[i] = modelDb["FEmembers"][i]["J"];
        A[i] = modelDb["FEmembers"][i]["A"];
        L[i] = modelDb["FEmembers"][i]["length"];
        G[i] = modelDb["FEmembers"][i]["G"]; //shear modulus

        // Calculate rotations
        double x1 = N[C[i][0]][0];
        double x2 = N[C[i][1]][0];
        double y1 = N[C[i][0]][1];
        double y2 = N[C[i][1]][1];
        double z1 = N[C[i][0]][2];
        double z2 = N[C[i][1]][2];
        double thetax = 0;                       // axial rotation, could be user-defined in future
        double thetay = atan2(z2 - z1, x2 - x1); // rotation in strong axis (think up-down on a standard I-beam)
        double thetaz = atan2(y2 - y1, x2 - x1); // rotation in weak axis (left-right)

        Rx[i] << 1, 0, 0, 0, cos(thetax), -sin(thetax), 0, sin(thetax), cos(thetax);
        Ry[i] << cos(thetay), 0, sin(thetay), 0, 1, 0, -sin(thetay), 0, cos(thetay);
        Rz[i] << cos(thetaz), -sin(thetaz), 0, sin(thetaz), cos(thetaz), 0, 0, 0, 1;
    }

    // TODO: Check data is correct and ready to process
    // check for no zero length things
    // Check for zero Jacobian derivatives at each quadrature point in each element
    for (int i = 0; i < nMembers; i++) {
        K_local.resize(0, 0);
        K_local.resize(12, 12);
        double e, a, l, ix, iy, jj, g;
        g = G[i];
        e = E[i];
        a = A[i];
        l = L[i];
        ix = Ix[i];
        iy = Iy[i];
        jj = J[i];

        K_local << e * a / l, 0, 0, 0, 0, 0, -e * a / l, 0, 0, 0, 0, 0,
            0, 12 * e * ix / l / l / l, 0, 0, 0, 6 * e * ix / l / l, 0, 12 * e * ix / l / l / l, 0, 0, 0, 6 * e * ix / l / l,
            0, 0, 12 * e * iy / l / l / l, 0, -6 * e * iy / l / l, 0, 0, 0, -12 * e * iy / l / l / l, 0, -6 * e * iy / l / l, 0,
            0, 0, 0, g* jj / l, 0, 0, 0, 0, 0, -g * jj / l, 0, 0,
            0, 0, -6 * e * iy / l / l, 0, 4 * e * iy / l, 0, 0, 0, 6 * e * iy / l / l, 0, 2 * e * iy / l, 0,
            0, 6 * e * ix / l / l, 0, 0, 0, 4 * e * ix / l, 0, -6 * e * ix / l / l, 0, 0, 0, 2 * e * ix / l,
            -e * a / l, 0, 0, 0, 0, 0, e* a / l, 0, 0, 0, 0, 0,
            0, -12 * e * ix / l / l / l, 0, 0, 0, -6 * e * ix / l / l, 0, 12 * e * ix / l / l / l, 0, 0, 0, -6 * e * ix / l / l,
            0, 0, -12 * e * iy / l / l / l, 0, 6 * e * iy / l / l, 0, 0, 0, 12 * e * iy / l / l / l, 0, 6 * e * iy / l / l, 0,
            0, 0, 0, -g * jj / l, 0, 0, 0, 0, 0, g* jj / l, 0, 0,
            0, 0, -6 * e * iy / l / l, 0, 2 * e * iy / l, 0, 0, 0, 6 * e * iy / l / l, 0, 4 * e * iy / l, 0,
            0, 6 * e * ix / l / l, 0, 0, 0, 2 * e * ix / l, 0, -6 * e * ix / l / l, 0, 0, 0, 4 * e * ix / l;

        Matrix3d R = Rz[i] * Ry[i] * Rx[i];
        Matrix3d zeros = Matrix3d::Zero(3, 3);
        rotationMatrices[i] << R, zeros, zeros, zeros, zeros, R, zeros, zeros, zeros, zeros, R, zeros, zeros, zeros, zeros, R;
        K_rotated_local[i] = rotationMatrices[i].transpose() * K_local * rotationMatrices[i];
        if (log(VERBOSE)) {
            std::cout << std::endl << "Rotated local stiffness matrix [" << i << "]:" << K_rotated_local[i] << std::endl;
        }
    }
    return K_rotated_local;
}

MatrixXd assembleGlobalStiffnessMatrix(std::vector<MatrixXd> K_rotated_local, std::vector<std::vector<int> >C, int nNodes, int nMembers) {
    // https://www.youtube.com/watch?v=vmjPL33Gugo&list=PLQVMpQ7G7XvHrdHLJgH8SeZQsiy2lQUcV&index=48
    if (log(INFO)) {
        std::cout << std::endl << "Assembling global stiffness matrix..." << std::endl;
    }
    MatrixXd K = MatrixXd::Zero(nNodes * 6, nNodes * 6);

    // There must be some optimization here.
    for (int e = 0; e < nMembers; e++) { //for each element
        for (int i = 0; i < 12; i++) { //for each row in local k
            for (int j = 0; j < 12; j++) { //for each column in local k
                int p = C[e][0]; //start node num
                int q = C[e][0];
                int offseti = 0;
                int offsetj = 0;
                if (i > 5) {
                    p = C[e][1]; //end node num
                    offseti = 6;
                }
                if (j > 5) {
                    q = C[e][1]; //end node num
                    offsetj = 6;
                }
                K(p * 6 + i - offseti, q * 6 + j - offsetj) += K_rotated_local[e](i, j);
            }
        }
    }
    if (log(VERBOSE)) {
        std::cout << "Global Stiffness Matrix:" << std::endl << K << std::endl;
    }
    return K;
}

VectorXd assembleGlobalForceVector(json modelDb, std::vector<std::vector<int> > C, int nMembers, int nNodes, std::vector<MatrixXd> rotationMatrices) {
    if (log(INFO)) {
        std::cout << std::endl << "Assembling global force vector..." << std::endl;
    }
    //for each ELEMENT
    // if udl, fl=ql[0,1/2,0, 0,L/12,0,   0,1/2,0, 0,-L/12]
    // I did not     ^  ^  ^  ^  ^   ^    ^  ^  ^  ^  ^   ^
    // calculate    Fx Fy  Fz T  My  Mz  Fx  Fy Fz T  My  Mz
    // this, but i  ---local-node-1----  ---local-node-2----
    // found it in a
    // book for 2d elems
    // and assumed
    // APPLiED torsion and axial is zero??
    std::vector<VectorXd> fl_local(nMembers, VectorXd(12));
    std::vector<VectorXd> fl_local_rotated(nMembers, VectorXd(12)); // aka [F]
    VectorXd fl_global = VectorXd::Zero(nNodes * 6);
    for (int i = 0; i < nMembers; i++) {
        double l = modelDb["FEmembers"][i]["length"];
        double qx = 0;//modelDb["FEmembers"][i]["udl"][0]; // should always be zero anyway
        double qy = modelDb["FEmembers"][i]["udl"][1];
        double qz = modelDb["FEmembers"][i]["udl"][2];
        fl_local[i] << qx * l / 2,
            qy* l / 2,
            qz* l / 2,
            0,
            qz* l* l / 24,
            qy* l* l / 24,
            qx* l / 2,
            qy* l / 2,
            qz* l / 2,
            0,
            -qz * l * l / 24,
            -qy * l * l / 24;
        fl_local_rotated[i] = rotationMatrices[i] * fl_local[i];
        // find node numbers
        int node1 = modelDb["FEmembers"][i]["from"];
        int node2 = modelDb["FEmembers"][i]["to"];
        fl_local_rotated[i][0] += (double)modelDb["FEnodes"][node1]["forces"]["x"];
        fl_local_rotated[i][1] += (double)modelDb["FEnodes"][node1]["forces"]["y"];
        fl_local_rotated[i][2] += (double)modelDb["FEnodes"][node1]["forces"]["z"];
        fl_local_rotated[i][3] += (double)modelDb["FEnodes"][node1]["moments"]["x"];
        fl_local_rotated[i][4] += (double)modelDb["FEnodes"][node1]["moments"]["y"];
        fl_local_rotated[i][5] += (double)modelDb["FEnodes"][node1]["moments"]["z"];
        fl_local_rotated[i][6] += (double)modelDb["FEnodes"][node2]["forces"]["x"];
        fl_local_rotated[i][7] += (double)modelDb["FEnodes"][node2]["forces"]["y"];
        fl_local_rotated[i][8] += (double)modelDb["FEnodes"][node2]["forces"]["z"];
        fl_local_rotated[i][9] += (double)modelDb["FEnodes"][node2]["moments"]["x"];
        fl_local_rotated[i][10] += (double)modelDb["FEnodes"][node2]["moments"]["y"];
        fl_local_rotated[i][11] += (double)modelDb["FEnodes"][node2]["moments"]["z"];

        if (log(VERBOSE))
            std::cout << std::endl << "fl_local[" << i << "]: " << std::endl << fl_local[i] << std::endl;
        if (log(VERBOSE))
            std::cout << std::endl << "fl_local_rotated[" << i << "]: " << std::endl << fl_local_rotated[i] << std::endl;
        if (log(VERBOSE))
            std::cout << std::endl << "rotation matrix[" << i << "]: " << std::endl << rotationMatrices[i] << std::endl;
    }

    // then assemble global fl by putting the forces where they
    // belong according to C. Care for rotations.
    for (int e = 0; e < nMembers; e++) { //for each element
        for (int i = 0; i < 12; i++) {                    //for each row in local fl_local_rotated
            int p = C[e][0]; //start node num
            int offseti = 0;
            if (i > 5) {
                p = C[e][1]; //end node num
                offseti = 6;
            }
            fl_global(p * 6 + i - offseti) += fl_local_rotated[e](i);
        }
    }

    if (log(VERBOSE))
        std::cout << std::endl << "Global Force vector:" << std::endl << fl_global << std::endl;

    return fl_global;
}

VectorXd assembleGlobalDisplacementVector(int nNodes, json modelDb, std::vector<int>& knowns_fd) {
    // Returns the global displacement vector and inserts the indices of known displacements to knowns_fd
    // knowns_fd must be an empty vector with 0 items

    if (log(INFO)) {
        std::cout << std::endl << "Assembling global displacement vector..." << std::endl;
    }

    // THE GLOBAL FORCE VECTOR MUST BE ADDED TO A VECTOR OF UNKNOWNS (fb)TO GET THE FINAL FORCE VECTOR
    // K * fd = fl+fb
    VectorXd fd_global(nNodes * 6);
    for (int i = 0; i < nNodes; i++) {
        //if known (nodes[i])
        if (modelDb["FEnodes"][i]["constraints"]["x"])
        {
            knowns_fd.push_back(i * 6);
            fd_global(i * 6) = 0; //modelDb["FEnodes"][i]["coords"][0];
        }
        if (modelDb["FEnodes"][i]["constraints"]["y"])
        {
            knowns_fd.push_back(i * 6 + 1);
            fd_global(i * 6 + 1) = 0; //modelDb["FEnodes"][i]["coords"][1];
        }
        if (modelDb["FEnodes"][i]["constraints"]["z"])
        {
            knowns_fd.push_back(i * 6 + 2);
            fd_global(i * 6 + 2) = 0; //modelDb["FEnodes"][i]["coords"][2];
        }
        if (modelDb["FEnodes"][i]["constraints"]["xx"])
        {
            knowns_fd.push_back(i * 6 + 3);
            fd_global(i * 6 + 3) = 0; // check this if correct, i have no mathematical knowledge about this
        }
        if (modelDb["FEnodes"][i]["constraints"]["yy"])
        {
            knowns_fd.push_back(i * 6 + 4);
            fd_global(i * 6 + 4) = 0; // check this if correct, i have no mathematical knowledge about this
        }
        if (modelDb["FEnodes"][i]["constraints"]["zz"])
        {
            knowns_fd.push_back(i * 6 + 5);
            fd_global(i * 6 + 5) = 0; // check this if correct, i have no mathematical knowledge about this
        }
    }
    if (log(VERBOSE))
        std::cout << std::endl << "Global Displacement vector:" << fd_global << std::endl;
    return fd_global;
}

VectorXd solveFEA(MatrixXd K, VectorXd fl_global, std::vector<int> knowns_fd, VectorXd fd) {
    if (log(INFO)) {
        std::cout << std::endl << "Solving FE equation..." << std::endl;
    }
    //K*q=fl_global
    // im using penalty method
    // penalty method: https://web.iitd.ac.in/~hegde/fem/lecture/lecture9.pdf
    // alternative: Sort the rows of everything so that the top part is unknowns using a permutation matrix or penalty method
    // anyway see https://math.stackexchange.com/questions/3180714/solve-system-of-linear-equations-ax-b-with-x-and-b-partially-known
    double large_stiffness = std::max(K.maxCoeff(), -K.minCoeff()) * 10000; // large stiffness
    for (std::vector<int>::iterator it = knowns_fd.begin(); it != knowns_fd.end(); it++) {
        int index = *it;
        double known_displacement = fd(index); // known displacement from BCs
        K(index, index) += large_stiffness;
        fl_global(index) += large_stiffness * known_displacement;
    }
    VectorXd q = K.lu().solve(fl_global);
    if (log(VERBOSE))
        std::cout << "Nodal displacements and roations:" << std::endl << q << std::endl;
    return q;

}

void saveJSON(json modelDb, VectorXd q, int nNodes, std::string in_filename) {
    // Store results to JSON file
    for (int i = 0; i < nNodes; i++) {
        double x = q(i * 6);
        double xx = modelDb["FEnodes"][i]["coords"][0];
        double xxx = x + xx;
        double y = q(i * 6 + 1);
        double yy = modelDb["FEnodes"][i]["coords"][1];
        double yyy = y + yy;
        double z = q(i * 6 + 2);
        double zz = modelDb["FEnodes"][i]["coords"][2];
        double zzz = z + zz;
        json node_res = {
            {"x", xxx},
            {"y", yyy},
            {"z", zzz},
            {"rx", q(i * 6 + 3)},
            {"ry", q(i * 6 + 4)},
            {"rz", q(i * 6 + 5)},
            {"dx",x},
            {"dy",y},
            {"dz",z} };
        modelDb["node_results"].push_back(node_res);
    }

    size_t lastindex = in_filename.find_last_of(".");
    std::string filename = std::string(in_filename).substr(0, lastindex) + "_res.json";
    if (log(INFO))
        std::cout << "Writing file \"" << filename << "\"...";
    std::ofstream ofile(filename);
    if (ofile.is_open()) {
        ofile << modelDb.dump(4);
    } else {
        if (log(ERRORS))
            std::cout << "Error writing to file." << std::endl;
        exit(1);
    }
    if (log(INFO))
        std::cout << " Success." << std::endl;

}

std::vector<std::vector<int> > makeDiaphragms(json modelDb, int nNodes) {
    // Makes a vector of vectors e.g diaphragm[i][j] have the index of the jth node of the ith diaphragm.
    double threshold = 0.5;
    std::vector<std::vector<int> > diaphragms(1);
    std::vector<double> zCoords;
    zCoords.push_back(0);
    for (int i = 0;i < nNodes;i++) {
        bool found = false;
        double currentNodeZ = modelDb["FEnodes"][i]["coords"][2];
        for (int j = 0;j < zCoords.size();j++) {
            // if node z within treshold of another node
            if (((currentNodeZ - threshold) < zCoords[j]) &&
                ((currentNodeZ + threshold) > zCoords[j])) {
                diaphragms[j].push_back(i);
                found = true;
                break;
            }
        }if (!found) {
            std::vector<int> t(1);
            t[0] = i;
            zCoords.push_back(currentNodeZ);
            diaphragms.push_back(t);
        }
    }
    return diaphragms;
}

std::vector<std::vector<double> > calculateCOR(json modelDb,
    std::vector<std::vector<int> >diaphragms,
    std::vector<std::vector<int> > C,
    int nMembers,
    int nNodes,
    std::vector<MatrixXd> rotationMatrices,
    MatrixXd K_global, std::string fn)
    {
    std::vector<std::vector<double > > CORs(diaphragms.size());
    
    //Copy modeldb
    json modelDbWithoutForce = modelDb;
    
    //remove all forces from the model
    for (int i=0;i<nMembers;i++){
        modelDbWithoutForce["FENodes"][i]["forces"]["x"]=0;
        modelDbWithoutForce["FENodes"][i]["forces"]["y"]=0;
        modelDbWithoutForce["FENodes"][i]["forces"]["z"]=0;
        modelDbWithoutForce["FENodes"][i]["moments"]["xx"]=0;
        modelDbWithoutForce["FENodes"][i]["moments"]["yy"]=0;
        modelDbWithoutForce["FENodes"][i]["moments"]["zz"]=0;
    }

    for (int j = 1;j < diaphragms.size();j++) {// for each diaphragm, and skip the bottom one
        // Clear forces
        json modelDbWithForceX = modelDbWithoutForce;
        json modelDbWithForceY = modelDbWithoutForce;
        json modelDbWithForceZ = modelDbWithoutForce;
         
        // Apply forces in each direction separately
        modelDbWithForceX["FENodes"][diaphragms[j][0]]["forces"]["x"]=10000;
        modelDbWithForceY["FENodes"][diaphragms[j][0]]["forces"]["y"]=10000;
        modelDbWithForceZ["FENodes"][diaphragms[j][0]]["moments"]["xx"]=10000;

        // Make global Force Vector fl
        VectorXd fl_globalX = assembleGlobalForceVector(modelDbWithForceX, C, nMembers, nNodes, rotationMatrices);
        VectorXd fl_globalY = assembleGlobalForceVector(modelDbWithForceY, C, nMembers, nNodes, rotationMatrices);
        VectorXd fl_globalZ = assembleGlobalForceVector(modelDbWithForceZ, C, nMembers, nNodes, rotationMatrices);

        // Make global displacement vector fd and store index of known displacements (boundary conditions)
        std::vector<int> knowns_fdX;
        std::vector<int> knowns_fdY;
        std::vector<int> knowns_fdZ;
        VectorXd fdX = assembleGlobalDisplacementVector(nNodes, modelDbWithForceX, knowns_fdX);
        VectorXd fdY = assembleGlobalDisplacementVector(nNodes, modelDbWithForceY, knowns_fdY);
        VectorXd fdZ = assembleGlobalDisplacementVector(nNodes, modelDbWithForceZ, knowns_fdZ);

        // Do math on K and fd and remove rows as needed to make distance between nodes of the same diaphragm to 
        // stay constant (= the same as the distances without any forces) after deformations of the structure.
        MatrixXd K_diaphragmed=K_global;
        

        // Solve FE equation
        VectorXd q = solveFEA(K_diaphragmed, fl_globalX, knowns_fdX, fdX);

        // Calculate rotation of diaphragm Rzx
        double Rzx;
        //----------------------------WRITE CODE HERE------------------------- 

        // Calculate Center of toraton (-Rzy/Rzz,Rzx/Rzz)
        //----------------------------WRITE CODE HERE------------------------- 
        //CORs.push_back(new vector<double>({centerX,centerY}) or some shit like this

        // save model for debugging purposes
        saveJSON(modelDbWithoutForce, q, nNodes, fn);
    }


}

int main(int argc, char* argv[]) {
    json modelDb;
    BNHParameters program_params;

    program_params = parseCLIArgs(argc, argv);

    modelDb = loadJSONFromFile(program_params.original_fn);
    int nMembers = modelDb["FEmembers"].size();
    int nNodes = modelDb["FEnodes"].size();

    // Convert json data to matrices
    std::vector<std::vector<double> > N = loadNodeMatrix(modelDb, nNodes);
    std::vector<std::vector<int> > C = loadConnectivityMatrix(modelDb, nMembers);

    // local stiffness matrices
    std::vector<MatrixXd> rotationMatrices(nMembers, MatrixXd(12, 12));
    std::vector<MatrixXd> K_rotated_local = loadRotatedLocalStiffnessMatrices(modelDb, nMembers, N, C, nNodes, rotationMatrices);

    // Make global stiffmess matrix
    MatrixXd K_global = assembleGlobalStiffnessMatrix(K_rotated_local, C, nNodes, nMembers);

    if (program_params.calcCOR) {
        // find which nodes are at the same floors
        std::vector<std::vector<int> > diaphragmNodes = makeDiaphragms(modelDb, nNodes);

        // Calculate rotations with set forces (https://wiki.csiamerica.com/display/etabs/Center+of+rigidity)
        std::vector<std::vector<double> > CORs = calculateCOR(modelDb, diaphragmNodes, C, nMembers, nNodes, rotationMatrices, K_global, "diap_x.json");
        //double Rzy = calculateRzy(modelDb, diaphragmNodes[i]);
        //double Rzz = calculateRzz(modelDb, diaphragmNodes[i]);
    }

    if (program_params.calcDisplacements) {
        // Make global Force Vector fl
        VectorXd fl_global = assembleGlobalForceVector(modelDb, C, nMembers, nNodes, rotationMatrices);

        // Make global displacement vector fd and store index of known displacements (boundary conditions)
        std::vector<int> knowns_fd;
        VectorXd fd = assembleGlobalDisplacementVector(nNodes, modelDb, knowns_fd);

        // Solve FE equation
        VectorXd q = solveFEA(K_global, fl_global, knowns_fd, fd);

        saveJSON(modelDb, q, nNodes, program_params.original_fn);
    }
}
