import numpy as np
import xlrd
import pandas as pd
from openpyxl import load_workbook

def read_xlsx_xlrd(path, cell):
    '''

    :param path: STR path of xlsx file
    :param cell: NDARRAY target location of cell
    :return: STR contents of cell
    '''
    chart = xlrd.open_workbook(path)
    first_sheet = chart.sheet_by_index(0)
    cell_list = first_sheet.row_slice(rowx=cell[0], start_colx=cell[1], end_colx = cell[2])
    cell_array = np.asarray(cell_list)
    cell_string = ''.join(map(str, cell_array))
    # eliminate all other redundant simbol
    tmp = cell_string.replace('text:\'', '')
    tmp = tmp.replace('text:\"', '')
    tmp = tmp.replace('"', '')
    tmp = tmp.replace(',', '')
    tmp = tmp.replace('.', '')
    cell_final = tmp.replace('\'', '')
    return cell_final


def read_csv(path):
    source = pd.read_csv(path, encoding = "ISO-8859-1")
    return source

def read_csv_tag(path, tag):
    '''

    :param path: directory of csv file
    :param tag: STR: tag name
    :return: STR: whole column of that tag
    '''
    content = pd.read_csv(path, encoding = "ISO-8859-1")
    return content[tag]


if __name__ == '__main__':
    path = 'Data/Competency_model_2_dimensional.csv'
    source = read_csv('Data/Competency_model_2_dimensional.csv')
    df = pd.read_csv("Data/Competency_model_2_dimensional.csv", encoding="ISO-8859-1")
    #df_skill_name.columns.values
    #print(source)
    content = read_csv_tag(path, tag = 'Skilled')
    #tmp = df['Competence'][0]
    print(content[0])
