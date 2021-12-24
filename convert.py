import os

from eppy.modeleditor import IDF

if __name__ == "__main__":
    IDD_PATH = os.path.normpath('./idd/V8-3-0-Energy+.idd')
    IDF.setiddname(IDD_PATH)

    idf = IDF('./convert.idf')
    idf.saveas('./convert_result.idf')
