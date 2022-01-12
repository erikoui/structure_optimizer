# Opens and displays a R2010 dxf file

import sys
import matplotlib.pyplot as plt

from ezdxf import recover, DXFStructureError
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# Safe loading procedure (requires ezdxf v0.14):
try:
    doc, auditor = recover.readfile('test.dxf')
except IOError:
    print(f'Not a DXF file or a generic I/O error.')
    sys.exit(1)
except DXFStructureError:
    print(f'Invalid or corrupted DXF file.')
    sys.exit(2)

# The auditor.errors attribute stores severe errors,
# which may raise exceptions when rendering.
if auditor.has_errors:
    exit()

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ctx = RenderContext(doc)
out = MatplotlibBackend(ax)
Frontend(ctx, out).draw_layout(doc.modelspace(), finalize=True)

plt.show()


# Save DXF (include .dxf in filename)
def save_dxf(ezdxf_doc,filename):
    print('Saving dxf to ' + filename)
    ezdxf_doc.saveas(filename)

#save PNG (inglude .png in filename)
def save_dxf_to_image(mpl_figure,filename):
    print('Saving image to ' + filename)
    mpl_figure.savefig(filename, dpi=300)

