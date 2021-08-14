import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import percentileofscore
import seaborn as sns
from millify import millify, prettify
import io
import base64
import matplotlib.pyplot as plt
import zipfile
from os import path

import scipy.stats as st
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

plt.switch_backend('Agg')

from re import DEBUG
from flask import Flask, render_template

app = Flask(__name__)


@app.route('/', methods=["POST", "GET"])
def index():

    if not path.exists('./data/train.pkl'):
        with zipfile.ZipFile('./data/train.pkl.zip', 'r') as zip_ref:
            zip_ref.extractall('./data/')

    dataFrame = readPickleFile()
    avg_age = int(get_avg(dataFrame, 'age'))
    avg_income = human_readable(get_avg(dataFrame, 'gross_income'))
    userCounts = get_count_values(dataFrame, 'segmentation')
    total_individual = human_readable(userCounts[0])
    total_student = human_readable(userCounts[1])
    total_vip = human_readable(userCounts[2])
    age_graph_url = age_distribution_graph_url(dataFrame)
    account_types_info = number_of_accounts(dataFrame)
    channels_url = number_of_channels_url(dataFrame)
    accounts = account_types_info[0]
    gender_url = gender_count_url(dataFrame)
    return render_template('dashboard.html',
                           avg_income=avg_income,
                           avg_age=avg_age,
                           total_vip=total_vip,
                           total_individual=total_individual,
                           total_student=total_student,
                           age_graph_url=age_graph_url,
                           channels_url=channels_url,
                           accounts=accounts,
                           account_len=len(accounts),
                           active_accounts=account_types_info[1],
                           gender_url=gender_url)


def loadDataUsingConfig():
    import utility as util
    # Read config file
    config_data = util.read_config_file("configuration.yaml")

    # read the file using config file
    file_type = config_data['file_type']
    source_file = "./data/" + config_data['file_name'] + f'.{file_type}'

    pd.set_option('display.max_columns', None)

    #print("",source_file)
    df = pd.read_csv(source_file, config_data['inbound_delimiter'])

    # rename columns
    df.rename(columns={
        'fecha_dato': 'table_partition',
        'ncodpers': 'customer_code',
        'ind_empleado': 'employee_index',
        'pais_residencia': 'customer_country_residence',
        'sexo': 'gender',
        'age': 'age',
        'fecha_alta': 'first_holder_contract_date',
        'ind_nuevo': 'new_customer_index',
        'antiguedad': 'customer_seniority',
        'indrel': 'primary_customer',
        'indrel_1mes': 'customer_type_beginning_month',
        'tiprel_1mes': 'customer_relation_type_beginning_month',
        'indresi': 'residence_index',
        'indext': 'foreigner_index',
        'canal_entrada': 'channel_used_to_join',
        'indfall': 'deceased_index',
        'tipodom': 'address_type',
        'cod_prov': 'province_code',
        'nomprov': 'province_name',
        'ind_actividad_cliente': 'activity_index',
        'renta': 'gross_income',
        'segmento': 'segmentation',
        'ind_ahor_fin_ult1': 'saving_account',
        'ind_aval_fin_ult1': 'guarantees',
        'ind_cco_fin_ult1': 'current_accounts',
        'ind_cder_fin_ult1': 'derivada_account',
        'ind_cno_fin_ult1': 'payroll_account',
        'ind_ctju_fin_ult1': 'junior_account',
        'ind_dela_fin_ult1': 'long_term_deposits',
        'ind_ecue_fin_ult1': 'e_account',
        'ind_fond_fin_ult1': 'funds',
        'ind_hip_fin_ult1': 'mortgage',
        'ind_plan_fin_ult1': 'pensions',
        'ind_pres_fin_ult1': 'loans',
        'ind_reca_fin_ult1': 'taxes',
        'ind_tjcr_fin_ult1': 'credit_card',
        'ind_valo_fin_ult1': 'securities',
        'ind_viv_fin_ult1': 'home_account',
        'ind_nomina_ult1': 'payroll',
        'ind_nom_pens_ult1': 'pensions_2',
        'ind_recibo_ult1': 'direct_debit',
        'ind_ctma_fin_ult1': 'medium_term_deposit',
        'ind_ctop_fin_ult1': 'long_term_deposit',
        'ind_ctpp_fin_ult1': 'particular_plus_account',
        'ind_deco_fin_ult1': 'short_term_deposit',
        'ind_deme_fin_ult1': 'medium_term_deposit_2',
        'ind_nuevo_ismissing': 'ind_nuevo_ismissing',
        'indrel_ismissing': 'indrel_ismissing',
        'tipodom_ismissing': 'tipodom_ismissing',
        'cod_prov_ismissing': 'cod_prov_ismissing',
        'ind_actividad_cliente_ismissing': 'ind_actividad_cliente_ismissing',
        'renta_ismissing': 'renta_ismissing',
        'ind_nomina_ult1_ismissing': 'ind_nomina_ult1_ismissing',
        'ind_nom_pens_ult1_ismissing': 'ind_nom_pens_ult1_ismissing'
    },
              inplace=True)

    # remove NA from age and gross
    # replace NA value in age with 0
    df['age'] = df['age'].replace({' NA': np.nan})
    df['age'] = df['age'].replace({' ': np.nan})
    df['age'] = df['age'].fillna(0)
    df["age"] = pd.to_numeric(df["age"])

    df['gross_income'] = df['gross_income'].replace({'nan': np.nan})
    df['gross_income'] = df['gross_income'].fillna(0)
    df["gross_income"] = pd.to_numeric(df["gross_income"])

    return df


def readPickleFile():
    df = pd.read_pickle('./data/train.pkl')
    # rename columns
    df.rename(columns={
        'fecha_dato': 'table_partition',
        'ncodpers': 'customer_code',
        'ind_empleado': 'employee_index',
        'pais_residencia': 'customer_country_residence',
        'sexo': 'gender',
        'age': 'age',
        'fecha_alta': 'first_holder_contract_date',
        'ind_nuevo': 'new_customer_index',
        'antiguedad': 'customer_seniority',
        'indrel': 'primary_customer',
        'indrel_1mes': 'customer_type_beginning_month',
        'tiprel_1mes': 'customer_relation_type_beginning_month',
        'indresi': 'residence_index',
        'indext': 'foreigner_index',
        'canal_entrada': 'channel_used_to_join',
        'indfall': 'deceased_index',
        'tipodom': 'address_type',
        'cod_prov': 'province_code',
        'nomprov': 'province_name',
        'ind_actividad_cliente': 'activity_index',
        'renta': 'gross_income',
        'segmento': 'segmentation',
        'ind_ahor_fin_ult1': 'saving_account',
        'ind_aval_fin_ult1': 'guarantees',
        'ind_cco_fin_ult1': 'current_accounts',
        'ind_cder_fin_ult1': 'derivada_account',
        'ind_cno_fin_ult1': 'payroll_account',
        'ind_ctju_fin_ult1': 'junior_account',
        'ind_dela_fin_ult1': 'long_term_deposits',
        'ind_ecue_fin_ult1': 'e_account',
        'ind_fond_fin_ult1': 'funds',
        'ind_hip_fin_ult1': 'mortgage',
        'ind_plan_fin_ult1': 'pensions',
        'ind_pres_fin_ult1': 'loans',
        'ind_reca_fin_ult1': 'taxes',
        'ind_tjcr_fin_ult1': 'credit_card',
        'ind_valo_fin_ult1': 'securities',
        'ind_viv_fin_ult1': 'home_account',
        'ind_nomina_ult1': 'payroll',
        'ind_nom_pens_ult1': 'pensions_2',
        'ind_recibo_ult1': 'direct_debit',
        'ind_ctma_fin_ult1': 'medium_term_deposit',
        'ind_ctop_fin_ult1': 'long_term_deposit',
        'ind_ctpp_fin_ult1': 'particular_plus_account',
        'ind_deco_fin_ult1': 'short_term_deposit',
        'ind_deme_fin_ult1': 'medium_term_deposit_2',
        'ind_nuevo_ismissing': 'ind_nuevo_ismissing',
        'indrel_ismissing': 'indrel_ismissing',
        'tipodom_ismissing': 'tipodom_ismissing',
        'cod_prov_ismissing': 'cod_prov_ismissing',
        'ind_actividad_cliente_ismissing': 'ind_actividad_cliente_ismissing',
        'renta_ismissing': 'renta_ismissing',
        'ind_nomina_ult1_ismissing': 'ind_nomina_ult1_ismissing',
        'ind_nom_pens_ult1_ismissing': 'ind_nom_pens_ult1_ismissing'
    },
              inplace=True)

    # remove NA from age and gross
    # replace NA value in age with 0
    df['age'] = df['age'].replace({' NA': np.nan})
    df['age'] = df['age'].replace({' ': np.nan})
    df['age'] = df['age'].fillna(0)
    df["age"] = pd.to_numeric(df["age"])

    df['gross_income'] = df['gross_income'].replace({'nan': np.nan})
    df['gross_income'] = df['gross_income'].fillna(0)
    df["gross_income"] = pd.to_numeric(df["gross_income"])

    return df


def get_avg(dataFrame, feature):
    return dataFrame[feature].mean()


def get_count_values(dataFrame, feature):
    return dataFrame[feature].value_counts()


def human_readable(number):
    return millify(number, precision=2)


def age_distribution_graph_url(dataFrame):
    plt.figure(figsize=(8, 5))
    age_plt = sns.distplot(dataFrame['age'], kde=False, color='Red')
    age_plt.set_xlabel('Age', fontsize=15)
    age_plt.set_title('Age distribution', fontsize=20)
    return get_graph_url(ax=age_plt)


def gender_count_url(dataFrame):
    plt.figure(figsize=(8, 5))
    df = dataFrame
    # df = dataFrame["gender"].replace({"H": "MALE", "V": "FEMALE"})
    df['gender'] = df['gender'].replace({"H": "MALE", "V": "FEMALE"})
    gender_plt = sns.countplot(x='gender', data=dataFrame)
    gender_plt.set_xlabel('Gender', fontsize=15)
    gender_plt.set_title('Gender Count', fontsize=20)
    return get_graph_url(ax=gender_plt)


def number_of_channels_url(dataFrame):
    plt.figure(figsize=(8, 5))
    barplot = sns.countplot(
        x='channel_used_to_join',
        data=dataFrame,
        order=pd.value_counts(
            dataFrame['channel_used_to_join']).iloc[:10].index)
    barplot.set_xlabel('', fontsize=15)
    barplot.set_title('Channels Used to Join', fontsize=20)
    plt.xticks(rotation=90)
    return get_graph_url(ax=barplot)


def number_of_accounts(dataFrame):
    accounts = [
        'saving_account', 'guarantees', 'current_accounts', 'derivada_account',
        'payroll_account', 'junior_account', 'medium_term_deposit',
        'medium_term_deposit_2', 'long_term_deposit',
        'particular_plus_account', 'short_term_deposit', 'e_account', 'funds',
        'mortgage', 'pensions', 'pensions_2', 'loans', 'taxes', 'credit_card',
        'securities', 'home_account', 'payroll', 'direct_debit'
    ]
    active = [
        prettify(dataFrame['saving_account'].value_counts()[1]),
        prettify(dataFrame['guarantees'].value_counts()[1]),
        prettify(dataFrame['current_accounts'].value_counts()[1]),
        prettify(dataFrame['derivada_account'].value_counts()[1]),
        prettify(dataFrame['payroll_account'].value_counts()[1]),
        prettify(dataFrame['junior_account'].value_counts()[1]),
        prettify(dataFrame['medium_term_deposit'].value_counts()[1]),
        prettify(dataFrame['medium_term_deposit_2'].value_counts()[1]),
        prettify(dataFrame['long_term_deposit'].value_counts()[1]),
        prettify(dataFrame['particular_plus_account'].value_counts()[1]),
        prettify(dataFrame['short_term_deposit'].value_counts()[1]),
        prettify(dataFrame['e_account'].value_counts()[1]),
        prettify(dataFrame['funds'].value_counts()[1]),
        prettify(dataFrame['mortgage'].value_counts()[1]),
        prettify(dataFrame['pensions'].value_counts()[1]),
        prettify(dataFrame['pensions_2'].value_counts()[1]),
        prettify(dataFrame['loans'].value_counts()[1]),
        prettify(dataFrame['taxes'].value_counts()[1]),
        prettify(dataFrame['credit_card'].value_counts()[1]),
        prettify(dataFrame['securities'].value_counts()[1]),
        prettify(dataFrame['home_account'].value_counts()[1]),
        prettify(dataFrame['payroll'].value_counts()[1]),
        prettify(dataFrame['direct_debit'].value_counts()[1])
    ]
    return [accounts, active]


def get_graph_url(ax):
    img = io.BytesIO()
    ax.figure.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url


if __name__ == "__main__":
    app.run(debug=True)