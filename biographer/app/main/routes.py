from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
from datetime import datetime, timezone
from flask import render_template, flash, redirect, url_for, request, g, \
    current_app, send_from_directory,request_finished, request_started
from flask_login import current_user, login_required
import sqlalchemy as sa
from app import db
import pingouin as pg
from app.main.forms import EditProfileForm
from app.models import User
from app.main import bp
from sklearn import linear_model
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.optimize import curve_fit
import os
from scipy.special import logit
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame, Series  # for convenience
import pims
import trackpy as tp
import math
import scipy as sp
from app.clear import clear_folders



@bp.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.now(timezone.utc)
        db.session.commit()




@bp.route('/')
def upload_form():
    clear_folders()
    return render_template('home.html')


@bp.route('/upload', methods=['POST','GET'])
@login_required
def upload_file():
    clear_folders()
    if 'file' not in request.files:
        return render_template('upload_form.html')
    global file
    file = request.files['file']
    if file.filename == '':
        flash(('No file selected!'))
        return redirect(url_for('main.upload_form'))
    if file:
        # Save the file to a temporary location
        global file_path
        file_path = os.path.join(current_app.root_path, 'uploads/'+ file.filename)
        file.save(file_path)
        
        # Read the Excel file
        global df
        df = pd.read_excel(file_path)
        global transformed_df
        transformed_df = df.copy()
        global graph
        graph = None
        # Render the template with the Excel data
       
        return render_template('display_excel.html', filename=file.filename, df=df.to_dict(orient='records'))


@bp.route('/normalize', methods=['POST','GET'])
def normalize():
    global transformed_df
    transformed_df=df.copy()
    global graph
    zero_percent = request.form.get('zero_percent')
    custom_input = request.form.get('custom_input')
    hcustom_input = request.form.get('hcustom_input')
    hundred_percent = request.form.get('hundred_percent')
    presentation = request.form.get('presentation')

    if zero_percent == 'smallest':
        zero_percent_values = df.min(axis=0)
    elif zero_percent == 'first':
        zero_percent_values = df.iloc[0]
    elif zero_percent == 'custom':
        zero_percent_values = pd.Series(float(custom_input), index=df.columns)
    elif zero_percent == 'sum':
        zero_percent_values = df.sum(axis=0)
    elif zero_percent == 'avg':
        zero_percent_values = df.mean(axis=0)
    
    if hundred_percent == "largest":
        hundred_percent_values = df.max(axis=0)
    elif hundred_percent == "last":
        hundred_percent_values = df.iloc[-1]
    elif hundred_percent == "hcustom":
        hundred_percent_values = pd.Series(float(hcustom_input), index=df.columns)
    elif hundred_percent == "hsum":
        hundred_percent_values = df.sum(axis=0)
    elif hundred_percent == "havg":
        hundred_percent_values = df.mean(axis=0)
    
    for column in df.columns:
        if presentation == "percentage":
            transformed_df[column] = ((df[column] - zero_percent_values[column]) / (hundred_percent_values[column] - zero_percent_values[column])) * 100
        elif presentation == "fraction":
            transformed_df[column] = (df[column] - zero_percent_values[column]) / (hundred_percent_values[column] - zero_percent_values[column])
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=transformed_df.iloc[:, 0], y=transformed_df.iloc[:, 1])
    global graph
    graph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_norm'+'.png'
    plt.savefig(os.path.join(current_app.root_path, 'static/'+ graph))
    plt.clf()          
    return render_template('display_excel.html', filename=file.filename, graph=graph,df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))

@bp.route('/transform', methods=['POST','GET'])
def transform():
    global transformed_df
    global graph
    biochem_transform = request.form.get('biochem')
    function = request.form.get('function')
    hill_input = request.form.get('hill_input')
    user_input = request.form.get('user_input')
    if biochem_transform == "eadie-hofstee":
        transformed_df.iloc[:, 0] = df.iloc[:, 1] / df.iloc[:, 0]
    elif biochem_transform == "hanes-woolf":
        transformed_df.iloc[:, 1] = df.iloc[:, 0] / df.iloc[:, 1]
    elif biochem_transform == "hill":
        transformed_df.iloc[:, 0] = np.log10(df.iloc[:, 0])
        transformed_df.iloc[:, 1] = np.log10(df.iloc[:, 1] / (float(hill_input) - df.iloc[:, 1]))
    elif biochem_transform == "log-log":
        transformed_df.iloc[:, 0] = np.log10(df.iloc[:, 0])
        transformed_df.iloc[:, 1] = np.log10(df.iloc[:, 1])
    elif biochem_transform == "scatchard":
        transformed_df.iloc[:, 1] = df.iloc[:, 1] / df.iloc[:, 0]
        transformed_df.iloc[:, 0] = df.iloc[:, 1]
    if function == "y*k":
        transformed_df.iloc[:, 1]=df.iloc[:, 1]*float(user_input)
    elif function == "y+k":
        transformed_df.iloc[:, 1]=df.iloc[:, 1]+float(user_input)
    elif function == "y-k":
        transformed_df.iloc[:, 1]=df.iloc[:, 1]-float(user_input)
    elif function == "y/k":
        transformed_df.iloc[:, 1]=df.iloc[:, 1]/float(user_input)
    elif function == "ysquared":
        transformed_df.iloc[:, 1]=df.iloc[:, 1]^2
    elif function == "y^k":
        transformed_df.iloc[:, 1]=df.iloc[:, 1]^float(user_input)
    elif function == "logy":
        transformed_df.iloc[:, 1]=np.log10(df.iloc[:, 1])
    elif function == "-logy":
        transformed_df.iloc[:, 1]=-1*np.log10(df.iloc[:, 1])
    elif function == "lny":
        transformed_df.iloc[:, 1]=np.log(df.iloc[:, 1])
    elif function == "10^y":
        transformed_df.iloc[:, 1]=10^df.iloc[:, 1]
    elif function == "e^y":
        transformed_df.iloc[:, 1]=math.exp(df.iloc[:, 1])
    elif function == "1/y":
        transformed_df.iloc[:, 1]=1/df.iloc[:, 1]
    elif function == "sqrty":
        transformed_df.iloc[:, 1]=math.sqrt(df.iloc[:, 1])
    elif function == "logity":
        transformed_df.iloc[:, 1]=logit(df.iloc[:, 1])
    elif function == "zscorey":
        transformed_df.iloc[:, 1]=stats.zscore(df.iloc[:, 1])
    elif function == "siny":
        transformed_df.iloc[:, 1]=math.sin(df.iloc[:, 1])
    elif function == "cosy":
        transformed_df.iloc[:, 1]=math.cos(df.iloc[:, 1])
    elif function == "tany":
        transformed_df.iloc[:, 1]=math.tan(df.iloc[:, 1])
    elif function == "arcsiny":
        transformed_df.iloc[:, 1]=math.asin(df.iloc[:, 1])
    elif function == "absy":
        transformed_df.iloc[:, 1]=abs(df.iloc[:, 1])
    elif function == "x/y":
        transformed_df.iloc[:, 1]=df.iloc[:, 0]/df.iloc[:, 1]
    elif function == "y/x":
        transformed_df.iloc[:, 1]=df.iloc[:, 1]/df.iloc[:, 0]
    elif function == "y-x":
        transformed_df.iloc[:, 1]=df.iloc[:, 1]-df.iloc[:, 0]
    elif function == "y+x":
        transformed_df.iloc[:, 1]=df.iloc[:, 1]+df.iloc[:, 0]
    elif function == "y*x":
        transformed_df.iloc[:, 1]=df.iloc[:, 1]*df.iloc[:, 0]
    elif function == "x-y":
        transformed_df.iloc[:, 1]=df.iloc[:, 0]-df.iloc[:, 1]
    elif function == "k-y":
        transformed_df.iloc[:, 1]=float(user_input)-df.iloc[:, 1]
    elif function == "k/y":
        transformed_df.iloc[:, 1]=float(user_input)/df.iloc[:, 1]
    elif function == "log2y":
        transformed_df.iloc[:, 1]=np.log2(df.iloc[:, 1])
    elif function == "2^y":
        transformed_df.iloc[:, 1]=2^df.iloc[:, 1]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=transformed_df.iloc[:, 0], y=transformed_df.iloc[:, 1])
    graph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_trans'+'.png'
    plt.savefig(os.path.join(current_app.root_path, 'static/'+ graph))
    plt.clf()          
    return render_template('display_excel.html', filename=file.filename, graph=graph,df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))

@bp.route('/transform-concentrations', methods=['POST','GET'])
def transform_concentrations():
    global transformed_df
    transformed_df=df.copy()
    global graph
    transform_concentrations = request.form.get('transformConcentration')
    userx_input = request.form.get('userx_input')
    if transform_concentrations == "changeX0":
        transformed_df.iloc[:, 0] = df.loc[df.iloc[:, 0] == 0, 'X'] = float(userx_input)
    elif transform_concentrations == "multConstant":
        transformed_df.iloc[:, 0]*=float(userx_input)
    elif transform_concentrations == "divConstant":
        transformed_df.iloc[:, 0]/=float(userx_input)
    elif transform_concentrations == "log10x":
        transformed_df.iloc[:, 0] = np.log10(df.iloc[:, 0])
    elif transform_concentrations == "lnx":
        transformed_df.iloc[:, 0] = np.log(df.iloc[:, 0])
    elif transform_concentrations == "log2x":
        transformed_df.iloc[:, 0] = np.log2(df.iloc[:, 0])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=transformed_df.iloc[:, 0], y=transformed_df.iloc[:, 1])
    graph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_transconc'+'.png'
    plt.savefig(os.path.join(current_app.root_path, 'static/'+ graph))
    plt.clf()          
    return render_template('display_excel.html', filename=file.filename, graph=graph,df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))

@bp.route('/area-under-curve', methods=['POST','GET'])
def area_under_curve():
    global transformed_df
    transformed_df=df.copy()
    global graph
    area = np.trapz(df.iloc[:, 1], x=df.iloc[:, 0])

    return render_template('display_excel.html', filename=file.filename, graph=graph,df=df.to_dict(orient='records'),  area=area,transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))

@bp.route('/fraction-of-total', methods=['POST','GET'])
def fraction_of_total():
    global transformed_df
    global graph
    transformed_df=df.copy()

    divfrac = request.form.get('divfrac')
    fracconfidence = request.form.get('fracconfidence')
    conf_input = request.form.get('conf_input')
    if divfrac == "column":
        transformed_df = df.div(df.sum(axis=0), axis=1)
    elif divfrac == "row":
        transformed_df = df.div(df.sum(axis=1), axis=0)
    elif divfrac == "grand":
        transformed_df = df.div(df.values.sum())
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=transformed_df.iloc[:, 0], y=transformed_df.iloc[:, 1])
    graph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_frac'+'.png'
    plt.savefig(os.path.join(current_app.root_path, 'static/'+ graph))
    plt.clf()

    
    if fracconfidence == "fracconfidence":
        lower_bound, upper_bound = calculate_confidence_intervals(df, float(conf_input))
        return render_template('display_excel.html', filename=file.filename, graph=graph, df=df.to_dict(orient='records'), lower_bound = lower_bound, upper_bound = upper_bound,  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
    
    return render_template('display_excel.html', filename=file.filename, graph=graph, df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))

@bp.route('/prune', methods=['POST','GET'])
def prune():
    global transformed_df
    global graph
    transformed_df=df.copy()

    pruneOption = request.form.get('pruneOption')
    avg_input = request.form.get('avg_input')
    if pruneOption == 'xRange':
        xmin = request.form.get('xmin')
        xmax = request.form.get('xmax')
        transformed_df= df[(df.iloc[:, 0] >= float(xmin)) & (df.iloc[:, 0] <= float(xmax))]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=transformed_df.iloc[:, 0], y=transformed_df.iloc[:, 1])
        graph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_prune'+'.png'
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ graph))
        plt.clf()

        return render_template('display_excel.html', filename=file.filename, graph=graph, df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
    elif pruneOption == 'average':
        transformed_df = df.groupby(df.index // float(avg_input)).mean()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=transformed_df.iloc[:, 0], y=transformed_df.iloc[:, 1])
        graph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_prune'+'.png'
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ graph))
        plt.clf()

        return render_template('display_excel.html', filename=file.filename, graph=graph, df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))

@bp.route('/transpose', methods=['POST','GET'])
def transpose():
    global transformed_df
    global graph
    transformed_df=df.transpose()
    plt.figure(figsize=(10, 6)) 
    sns.lineplot(data=transformed_df.T, dashes=False)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Transposed Data")
    plt.legend(title="Data Sets", labels=transformed_df.index)
    graph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_tps'+'.png'
    plt.savefig(os.path.join(current_app.root_path, 'static/'+ graph))
    plt.clf()
    return render_template('display_excel.html', filename=file.filename, graph=graph, df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))

@bp.route('/nominal-test', methods = ['POST','GET'])
def perform_nominal_test():
    global transformed_df
    global graph 
    test = request.form.get('test_nominal')
    if test == 'exact_test_of_goodness_of_fit':
        num_successes = request.form.get('success_input')  
        num_trials = request.form.get('trials_input')
        null_hypothesis_prob = request.form.get('prob')
        # Perform the binomial test
        p_value = stats.binomtest(int(num_successes), n=int(num_trials), p=float(null_hypothesis_prob))

        # You can then interpret the p-value to make a decision
        return render_template('display_excel.html', graph=graph, test_results={"Exact Test of Goodness-of-Fit": "Ran successfully."}, statistic = "p-value: " + str(p_value.pvalue),df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
    elif test == 'chi_square_test_of_goodness_of_fit':
        observed_data = transformed_df.iloc[:, 0]
        expected_data = transformed_df.iloc[:, 1]
        chi_square_test_statistic, p_value = stats.chisquare(observed_data,expected_data) 
        return render_template('display_excel.html', graph=graph, test_results={"Chi-Square Test of Goodness-of-Fit": "Ran successfully."}, statistic = "Chi Square Test Statistic: " + str(chi_square_test_statistic), stat2 = "p-value: " + str(p_value),df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
    elif test == 'chi_square_test_of_independence':
        chi2, p, dof, expected = stats.chi2_contingency(transformed_df)
        observed_proportions = []
        for i in range(transformed_df.shape[1]):
            observed_proportions.append(transformed_df.iloc[:, i].sum() / transformed_df.sum().sum())
        expected_table = pd.DataFrame(expected, index=transformed_df.index)
        categories = list(transformed_df.columns)
        plt.bar(range(len(categories)), observed_proportions, tick_label=categories)
        plt.xlabel('Categories')
        plt.ylabel('Proportions')
        plt.title('Observed Proportions')
        plt.xticks(rotation = 45)
        statsgraph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_csit'+'.png'
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ statsgraph))
        plt.clf()
        return render_template('display_excel.html', graph=graph, statsgraph = statsgraph,
                                test_results={"Chi-Square Test of Independence": "(p-value = " + str(p)+")"}, 
                                df=df.to_dict(orient='records'), 
                                
                                transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None),
                                statistic="Chi-square statistic: " + str(chi2),
                                dof="Degrees of freedom: " + str(dof),
                                expected_table = (expected_table.to_dict(orient='records') if df is not None else None)
                                )
    elif test == 'fishers_exact_test':
        odd_ratio, p_value = stats.fisher_exact(transformed_df)
        observed_proportions = []
        for i in range(transformed_df.shape[1]):
            observed_proportions.append(transformed_df.iloc[:, i].sum() / transformed_df.sum().sum())
        
        categories = list(transformed_df.columns)
        plt.bar(range(len(categories)), observed_proportions, tick_label=categories)
        plt.xlabel('Categories')
        plt.ylabel('Proportions')
        plt.title('Observed Proportions')
        plt.xticks(rotation = 45)
        statsgraph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_fish'+'.png'
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ statsgraph))
        plt.clf()
        return render_template('display_excel.html', graph=graph, statsgraph = statsgraph,
                                test_results={"Fisher's Exact Test": "(p-value = " + str(p_value)+")"}, 
                                df=df.to_dict(orient='records'), 
                                transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None),
                                statistic="Odd ratio " + str(odd_ratio),
                                )

@bp.route('/one-measurement-test',methods=['POST', 'GET'])
def perform_one_measurement_test():
    global transformed_df
    global graph
    test = request.form.get('test_one_measurement')
    if test == 'one_sample_t_test':
        test_mean = request.form.get('means_input')
        t_statistic, p_value = stats.ttest_1samp(a=transformed_df, popmean=float(test_mean))
        return render_template('display_excel.html', graph=graph, test_results={"One sample t-test": "(p-value = " + str(p_value)+")"}, statistic = "t-statistic is: " + str(t_statistic), df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
    elif test == 'two_sample_t_test':
        t_statistic, p_value = stats.ttest_ind(a=transformed_df.iloc[:, 0], b=transformed_df.iloc[:, 1], equal_var=True)
        return render_template('display_excel.html', graph=graph, test_results={"Two sample t-test": "(p-value = " + str(p_value)+")"}, statistic = "t-statistic is: " + str(t_statistic), df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
    elif test == 'homoscedasticity':
        statistic, p_value = stats.bartlett(transformed_df.iloc[:,0], transformed_df.iloc[:,1])
        return render_template('display_excel.html', graph=graph, test_results={"Bartlett's Test": "(p-value = " + str(p_value)+")"}, statistic = "Bartlett's test statistic is: " + str(statistic), df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
    elif test == 'one_way_anova':
        f_statistic, p_value = stats.f_oneway(*[transformed_df[col] for col in transformed_df.columns])
        categories = list(transformed_df.columns)
        means = transformed_df.mean()
        plt.bar(range(len(categories)), means.values, tick_label=categories)
        statsgraph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_anova'+'.png'
        plt.xticks(rotation = 45)
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ statsgraph))
        plt.clf()
        return render_template('display_excel.html', graph=graph, statsgraph = statsgraph, test_results={"One-way ANOVA": "(p-value = " + str(p_value)+")"}, statistic = "F-statistic is: " + str(f_statistic), df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
    elif test == 'kruskal_wallis_test':
        h_statistic, p_value = stats.kruskal(*[transformed_df[col] for col in transformed_df.columns])
        return render_template('display_excel.html', graph=graph, test_results={"Kruskal-Wallis": "(p-value = " + str(p_value)+")"}, statistic = "H-statistic is: " + str(h_statistic), df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
    elif test == 'two_way_anova':

        anova_table = pg.anova(data=transformed_df, dv=transformed_df.columns[2], between=[transformed_df.columns[0], transformed_df.columns[1]])
        plt.title('Two-Way ANOVA')
        sns.barplot(data=transformed_df, x=transformed_df.iloc[:, 0], y=transformed_df.iloc[:, 2], hue=transformed_df.iloc[:, 1], legend=True, palette='rocket')
        plt.xlabel(transformed_df.columns[0])
        plt.ylabel(transformed_df.columns[2])
        plt.legend(title=transformed_df.columns[1])
        statsgraph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_twowayanova'+'.png'
        plt.xticks(rotation = 45)
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ statsgraph))
        plt.clf()
        return render_template('display_excel.html', twoanova=anova_table.to_dict(orient='records'),graph=graph, statsgraph = statsgraph, test_results={"Two-way ANOVA": "Sucessfully ran."}, df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
        
    elif test == 'paired_t_test':
        before = transformed_df.iloc[:,0]
        after = transformed_df.iloc[:,1]
        t_statistic, p_value = stats.ttest_rel(before, after)
        labels = ['Before', 'After']
        means = [np.mean(before), np.mean(after)]
        errors = [np.std(before), np.std(after)]
        x_pos = np.arange(len(labels))
        plt.bar(x_pos, means, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
        plt.xticks(x_pos, labels)
        plt.ylabel('Mean Value')
        plt.title('Paired t-test Results')
        statsgraph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_pairt'+'.png'
        plt.xticks(rotation = 45)
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ statsgraph))
        plt.clf()
        return render_template('display_excel.html', graph=graph, statsgraph = statsgraph, test_results={"Paired t-test": "(p-value = " + str(p_value)+")"}, statistic = "t-statistic is: " + str(t_statistic), df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))

    elif test == 'wilcoxon_signed_rank_test':
        before = transformed_df.iloc[:,0]
        after = transformed_df.iloc[:,1]
        t_statistic, p_value = stats.wilcoxon(before, after)
        labels = ['Before', 'After']
        means = [np.mean(before), np.mean(after)]
        errors = [np.std(before), np.std(after)]
        x_pos = np.arange(len(labels))
        plt.bar(x_pos, means, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
        plt.xticks(x_pos, labels)
        plt.ylabel('Mean Value')
        plt.title('Paired t-test Results')
        statsgraph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_wilc'+'.png'
        plt.xticks(rotation = 45)
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ statsgraph))
        plt.clf()
        return render_template('display_excel.html', graph=graph, statsgraph = statsgraph, test_results={"Wilcoxon Signed Rank Test": "(p-value = " + str(p_value)+")"}, statistic = "t-statistic is: " + str(t_statistic), df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
        

@bp.route('/multiple-measurement-test',methods=['POST'])
def perform_multiple_measurement_test():
    global transformed_df
    global graph
    test = request.form.get('test_multiple_measurement')
    if test == 'linear_regression_and_correlation':
        X = transformed_df.iloc[:,0]
        Y = transformed_df.iloc[:,1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    
        # Plot the data and regression line
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X, y=Y, label='Data')
        sns.lineplot(x=X, y=slope*X + intercept, color='red', label='Linear Regression')
        plt.xlabel(transformed_df.columns[0])
        plt.ylabel(transformed_df.columns[1])
        plt.title("Simple Linear Regression")
        plt.legend()
        statsgraph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_linr'+'.png'
        plt.xticks(rotation = 45)
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ statsgraph))
        plt.clf()
        return render_template('display_excel.html', graph=graph, statsgraph = statsgraph, test_results={"Linear Regression": "Sucessfully ran."}, 
                               statistic="p-value: " + str(p_value), stat2 = 'r-value: ' + str(r_value**2), stat3 = 'Slope: ' + str(slope), stat4 = 'Intercept: ' + str(intercept),
                               stat5='Standard Error: ' + str(std_err),
                                 df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
    
    elif test == 'spearman_rank_correlation':
        X = transformed_df.iloc[:,0]
        Y = transformed_df.iloc[:,1]
        corr, pval = stats.spearmanr(X, Y)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X, y=Y, label='Data')
        plt.xlabel(transformed_df.columns[0])
        plt.ylabel(transformed_df.columns[1])
        plt.title("Spearman Rank Correlation")
        plt.legend()
        statsgraph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_spr'+'.png'
        plt.xticks(rotation = 45)
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ statsgraph))
        plt.clf()
        return render_template('display_excel.html', graph=graph, statsgraph = statsgraph, test_results={"Spearman Rank Correlation": "Sucessfully ran."}, 
                               statistic="p-value: " + str(pval), stat2 = 'Spearman\'s Correlation Coefficient: ' + str(corr), 
                               df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))

    elif test == 'polynomial_regression':
        x = transformed_df.iloc[:,0]
        y = transformed_df.iloc[:,1]
        mymodel = np.poly1d(np.polyfit(x, y, 3))
        myline = np.linspace(1, int(max(x)), int(max(y)))
        r_value = stats.linregress(x, y)
        plt.scatter(x, y)
        plt.plot(myline, mymodel(myline))
        plt.xlabel(transformed_df.columns[0])
        plt.ylabel(transformed_df.columns[1])
        plt.title("Polynomial Regression")
        statsgraph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_polr'+'.png'
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ statsgraph))
        plt.clf()
        return render_template('display_excel.html', graph=graph, statsgraph = statsgraph, test_results={"Polynomial Regression": "Sucessfully ran."}, 
                               statistic="r-squared: " + str(r2_score(y, mymodel(x))),
                               df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
         
    elif test == 'analysis_of_covariance':
        result = pg.ancova(data=transformed_df, dv=transformed_df.columns[2], covar=transformed_df.columns[1], between= transformed_df.columns[0])
        x = transformed_df.iloc[:,0]
        y = transformed_df.iloc[:,1]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y, label='Data')
        plt.xlabel(transformed_df.columns[0])
        plt.ylabel(transformed_df.columns[1])
        plt.title("ANCOVA")
        statsgraph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_ancova'+'.png'
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ statsgraph))
        plt.clf()
        return render_template('display_excel.html', twoanova=result.to_dict(orient='records'),graph=graph, statsgraph = statsgraph, test_results={"ANCOVA": "Sucessfully ran."}, df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))

    elif test == 'simple_logistic_regression':
        def logistic_function(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))
        X = transformed_df.iloc[:,0]
        Y = transformed_df.iloc[:,1]

        popt, pcov = curve_fit(logistic_function, X, Y)

        L, k, x0 = popt

        print("L (Maximum Value):", L)
        print("k (Steepness):", k)
        print("x0 (Midpoint):", x0)

        # Generate logistic regression curve
        X_curve = np.linspace(min(X), max(X), 100)
        Y_curve = logistic_function(X_curve, *popt)

        # Plot the data and logistic regression curve
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X, y=Y, label='Data')
        plt.plot(X_curve, Y_curve, color='red', label='Logistic Regression')
        plt.xlabel(transformed_df.columns[0])
        plt.ylabel(transformed_df.columns[1])
        plt.title("Simple Logistic Regression")
        plt.legend()
        statsgraph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_logr'+'.png'
        plt.xticks(rotation = 45)
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ statsgraph))
        plt.clf()
        return render_template('display_excel.html', graph=graph, statsgraph = statsgraph, test_results={"Logistic Regression": "Sucessfully ran."}, 
                               statistic="L (Maximum Value): " + str(L), stat2 = 'k (Steepness): ' + str(k), stat3 = 'x0 (Midpoint): ' + str(x0),
                                 df=df.to_dict(orient='records'),  transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))
    

    
        

@bp.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash(('Your changes have been saved.'))
        return redirect(url_for('main.edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title=('Edit Profile'),
                           form=form)

@bp.route('/user/<username>')
@login_required
def user(username):
    user = db.first_or_404(sa.select(User).where(User.username == username))
    return render_template('user.html', user=user)

@bp.route('/tool-list')
@login_required
def tool_list():
    return render_template('tools/tool_list.html')

@bp.route('/hemocytometer', methods=['GET','POST'])
@login_required
def hemocytometer():
    return render_template('tools/hemocytometer/hemocytometer.html', title='Hemocytometer')

@bp.route('/uploads/<path:filename>')
def get_upload(filename):
    return send_from_directory('uploads', filename)

@bp.route('/hemocytometer-upload', methods=['GET','POST'])
@login_required
def hemocytometer_upload():
    clear_folders()
    if 'img' not in request.files:
        return ('No file part')
    global img
    img = request.files['img']
    if img.filename == '':
        flash(('No image selected!'))
        return redirect(url_for('main.hemocytometer-upload'))
    if img:
        # Save the file to a temporary location
        global img_path
        img_path = os.path.join(current_app.root_path, 'uploads/'+ img.filename)
        img.save(img_path)
    return render_template('tools/hemocytometer/display_hemo.html', imgname=img.filename)

@bp.route('/count', methods=['POST'])
def count():
    mpl.rc('figure',  figsize=(10, 5))
    mpl.rc('image', cmap='gray')
    @pims.pipeline
    def gray(image):
        return image[:, :, 1]  # Take just the green channel
    frames = gray(pims.open(img_path))
    micron_per_pixel = 0.15192872980868
    feature_diameter = 2.12 # um
    radius = int(np.round(feature_diameter/2.0/micron_per_pixel))
    if radius % 2 == 0:
        radius += 1
    print('Using a radius of {:d} px'.format(radius))
    frames
    f_bf = tp.locate_brightfield_ring(frames[0], 2.0*radius+1)
    plt.figure()
    tp.annotate(f_bf, frames[0], plot_style={'markersize': radius*2}, ax=plt.gca())
    plt.xlabel('Number of cells: ' + str(len(f_bf)))
    countname=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'count'+ '.png'
    plt.savefig(os.path.join(current_app.root_path, 'static/'+ countname))
    return render_template('tools/hemocytometer/display_hemo.html', imgname=img.filename, countname=countname)

def wilson_brown_confidence_interval(n, p, alpha=0.05):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    center = p + z_alpha**2 / (2 * n)
    width = z_alpha * np.sqrt(p * (1 - p) / n + z_alpha**2 / (4 * n**2))
    lower_bound = (center - width) / (1 + z_alpha**2 / n)
    upper_bound = (center + width) / (1 + z_alpha**2 / n)
    return lower_bound, upper_bound

def calculate_confidence_intervals(data, percentage):
    if percentage > 0 and percentage < 100:
        n = data.sum().sum()  # Total number of observations
        p = data.values.sum() / (data.shape[0] * data.shape[1])  # Proportion of successes
        lower_bound, upper_bound = wilson_brown_confidence_interval(n, p)
        return lower_bound, upper_bound
    else:
        return None, None




@bp.route('/documentation-intro', methods=['GET','POST'])
def documentation_intro():
    return render_template('documentation/introduction.html')

@bp.route('/documentation-install', methods=['GET','POST'])
def documentation_install():
    return render_template('documentation/install.html')

@bp.route('/documentation-transform', methods=['GET','POST'])
def documentation_transform():
    return render_template('documentation/transform.html')

@bp.route('/documentation-tests-for-nominal-variables', methods=['GET','POST'])
def documentation_dnom():
    return render_template('documentation/dnom.html')

@bp.route('/documentation-tests-for-one-measurement-variable', methods=['GET','POST'])
def documentation_done():
    return render_template('documentation/done.html')

@bp.route('/documentation-tests-for-multiple-measurement-variables', methods=['GET','POST'])
def documentation_dmult():
    return render_template('documentation/dmult.html')

@bp.route('/documentation-hemocytometer', methods=['GET','POST'])
def documentation_hemo():
    return render_template('documentation/hemo.html')

@bp.route('/documentation-notebook', methods=['GET','POST'])
def documentation_notebook():
    return render_template('documentation/notebook.html')

@bp.route('/about', methods=['GET','POST'])
def about():
    return render_template('about.html')