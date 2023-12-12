import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
path = 'https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/'

d_clients = pd.read_csv(f'{path}/D_clients.csv')
d_target = pd.read_csv(f'{path}/D_target.csv')
d_last_credit = pd.read_csv(f'{path}/D_last_credit.csv')
d_job = pd.read_csv(f'{path}/D_job.csv')
d_salary = pd.read_csv(f'{path}/D_salary.csv')
d_work = pd.read_csv(f'{path}/D_work.csv')
d_loan = pd.read_csv(f'{path}/D_loan.csv')
d_close_loan = pd.read_csv(f'{path}/D_close_loan.csv')
d_pens = pd.read_csv(f'{path}/D_pens.csv')

d_salary_without_doubles = d_salary[~d_salary.duplicated()]
d_loan_by_client = d_loan.groupby(['ID_CLIENT'], as_index=False).agg(LOAN_NUM_TOTAL=('ID_LOAN', 'nunique'))
d_close_loan_by_client = d_loan.merge(d_close_loan[d_close_loan['CLOSED_FL'] == 1],
how = 'inner', on = 'ID_LOAN').groupby(['ID_CLIENT'], as_index=False).agg(LOAN_NUM_CLOSED=('ID_LOAN', 'nunique'))
d_loan_by_client = d_loan_by_client.merge(d_close_loan_by_client, how = 'left', on = 'ID_CLIENT').fillna(0).astype(int)

data = d_clients.merge(d_loan_by_client, how = 'left', left_on = 'ID', right_on = 'ID_CLIENT')
data = data.merge(d_target, how = 'left', on = 'ID_CLIENT')
data = data.merge(d_salary_without_doubles, how = 'left', on = 'ID_CLIENT')
data = data.merge(d_last_credit, how = 'left', on = 'ID_CLIENT')
data = data[['AGREEMENT_RK', 'TARGET', 'AGE', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
             'GENDER', 'CHILD_TOTAL', 'DEPENDANTS', 'PERSONAL_INCOME',  'FAMILY_INCOME', 'LOAN_NUM_TOTAL',
             'LOAN_NUM_CLOSED', 'CREDIT', 'TERM', 'FST_PAYMENT']]

data = data[['AGREEMENT_RK', 'TARGET', 'AGE', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
             'GENDER', 'MARITAL_STATUS', 'EDUCATION', 'CHILD_TOTAL', 'DEPENDANTS', 'PERSONAL_INCOME',  'FAMILY_INCOME',
             'LOAN_NUM_CLOSED', 'CREDIT', 'TERM', 'FST_PAYMENT']]

st.title("EDA")
st.subheader("Описательные статистики")
st.write(data.describe())
st.write("Категориальные переменные")
st.write(data.describe(include = 'object'))
st.subheader("Пример данных")
st.write(data.head(10))
st.subheader("Распределение целевой переменной")
fig, ax = plt.subplots()
sns.barplot(x = data['TARGET'].value_counts().index, y = data['TARGET'].value_counts().values, ax=ax)
ax.set_title('TARGET')
st.pyplot(fig)
numeric_columns  = ['AGE', 'CHILD_TOTAL', 'DEPENDANTS',
       'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED',
       'PERSONAL_INCOME', 'CREDIT', 'TERM', 'FST_PAYMENT']
titles  = ['Возраст', 'Количество детей', 'Количество иждивенцев',
       'Количество ссуд клиента', 'Количество погашенных ссуд клиента',
       'Личный доход', 'Сумма последнего кредита клиента', 'Срок кредита', 'Первоначальный взнос']
titles_dict = dict(zip(numeric_columns, titles))
st.subheader("Распределение независимых переменных")
for column in numeric_columns:
    fig, ax = plt.subplots()
    sns.histplot(data[column], ax=ax)
    ax.set_title(titles_dict[column])
    st.pyplot(fig)

cat_and_nominal_columns = ['SOCSTATUS_WORK_FL', 'MARITAL_STATUS']
cat_titles = ['Наличие работы', 'Семейное положение']
cat_titles_dict = dict(zip(cat_and_nominal_columns, cat_titles))
for column in cat_and_nominal_columns:
    fig, ax = plt.subplots()
    sns.barplot(x = data[column].value_counts().index, y = data[column].value_counts().values, ax=ax)
    ax.set_title(cat_titles_dict[column])
    plt.xticks(rotation=50)
    st.pyplot(fig)
fig, ax = plt.subplots()
sns.barplot(x = data['FAMILY_INCOME'].value_counts().index,
y = data['FAMILY_INCOME'].value_counts().values, ax=ax, order = ['до 5000 руб.', 'от 5000 до 10000 руб.',  'от 10000 до 20000 руб.',
'от 20000 до 50000 руб.', 'свыше 50000 руб.'])
plt.xticks(rotation=50)
st.pyplot(fig)

st.subheader("Диаграммы рассеяния")
column_pairs = [('CHILD_TOTAL', 'PERSONAL_INCOME'), ('CREDIT', 'PERSONAL_INCOME')
               , ('AGE', 'PERSONAL_INCOME'), ('PERSONAL_INCOME', 'LOAN_NUM_TOTAL')
               , ('PERSONAL_INCOME', 'LOAN_NUM_CLOSED')]
titles_scatter = ['Количество детей и доход', 'Сумма последнего кредита и личный доход',
                  'Возраст и личный доход', 'Личный доход и число ссуд клиента',
                   'Личный доход и число погашенных ссуд клиента']
titles_scatter_dict = dict(zip(column_pairs, titles_scatter))
for x_column, y_column in column_pairs:
    fig, ax = plt.subplots()
    sns.scatterplot(x=data[x_column], y=data[y_column], ax=ax)
    ax.set_title(titles_scatter_dict[(x_column, y_column)])
    st.pyplot(fig)
st.subheader("Матрица корреляций")
corr_matrix = data[numeric_columns].corr(numeric_only = True)
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=.5, fmt=".2f", ax=ax)
st.pyplot(fig)
