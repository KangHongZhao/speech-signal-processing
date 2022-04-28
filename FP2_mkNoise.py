"""
FP2_mkNoise (25%)
"""
## import modules needed below
from matplotlib.backends.backend_pdf import PdfPages as pdfs



def save2PDF(fname, figs):
    handle = pdfs(fname)
    try:
        for fig in figs:
            handle.savefig(fig)
    except:
        print("Errors have occurred while saving PDF!")
    finally:
        handle.close()
## insert your code below

