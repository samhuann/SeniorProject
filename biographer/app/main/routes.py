from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
from datetime import datetime, timezone
from flask import render_template, flash, redirect, url_for, request, g, \
    current_app, send_from_directory
from flask_login import current_user, login_required
import sqlalchemy as sa
from app import db
from app.main.forms import EditProfileForm
from app.models import User
from app.main import bp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress,zscore 
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

@bp.before_app_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.now(timezone.utc)
        db.session.commit()



@bp.route('/')
@login_required
def upload_form():
    return render_template('upload_form.html', title='Home')


@bp.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
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
        # Render the template with the Excel data
       
        return render_template('display_excel.html', filename=file.filename, tables=[df.to_html(classes='data')], titles=df.columns.values)


@bp.route('/nominal_test', methods=['POST'])
def perform_nominal_test():
    test = request.form['test_nominal']
    result = perform_nominal_test(test)
    return render_template('display_excel.html', test_results={test: result}, tables=[df.to_html(classes='data')], titles=df.columns.values)

@bp.route('/one_measurement_test', methods=['POST'])
def perform_one_measurement_test():
    test = request.form['test_one_measurement']
    result = perform_one_measurement_test(test)
    return render_template('display_excel.html', test_results={test: result}, tables=[df.to_html(classes='data')], titles=df.columns.values)

@bp.route('/multiple_measurement_test', methods=['POST'])
def perform_multiple_measurement_test():
    test = request.form['test_multiple_measurement']
    result = perform_multiple_measurement_test(test)
    return render_template('display_excel.html', test_results={test: result}, tables=[df.to_html(classes='data')], titles=df.columns.values)


# Placeholder functions for performing tests
def perform_nominal_test(test):
    if test == 'exact_test_of_goodness_of_fit':
        # Placeholder for performing Exact Test of Goodness-of-Fit
        return "Result of Exact Test of Goodness-of-Fit"
    elif test == 'power_analysis':
        # Placeholder for performing Power Analysis
        return "Result of Power Analysis"
    elif test == 'chi_square_test_of_goodness_of_fit':
        # Placeholder for performing Chi-Square Test of Goodness-of-Fit
        return "Result of Chi-Square Test of Goodness-of-Fit"
    elif test == 'g_test_of_goodness_of_fit':
        # Placeholder for performing G-Test of Goodness-of-Fit
        return "Result of G-Test of Goodness-of-Fit"
    elif test == 'chi_square_test_of_independence':
        # Placeholder for performing Chi-Square Test of Independence
        return "Result of Chi-Square Test of Independence"
    elif test == 'g_test_of_independence':
        # Placeholder for performing G-Test of Independence
        return "Result of G-Test of Independence"
    elif test == 'fishers_exact_test':
        # Placeholder for performing Fisher's Exact Test
        return "Result of Fisher's Exact Test"
    elif test == 'cochran_mantel_haenszel_test':
        # Placeholder for performing Cochran-Mantel-Haenszel Test
        return "Result of Cochran-Mantel-Haenszel Test"

def perform_one_measurement_test(test):
    if test == 'one_sample_t_test':
        # Placeholder for performing One-Sample t-Test
        return "Result of One-Sample t-Test"
    elif test == 'two_sample_t_test':
        # Placeholder for performing Two-Sample t-Test
        return "Result of Two-Sample t-Test"
    elif test == 'homoscedasticity':
        # Placeholder for performing Homoscedasticity test
        return "Result of Homoscedasticity test"
    elif test == 'one_way_anova':
        # Placeholder for performing One-Way ANOVA
        return "Result of One-Way ANOVA"
    elif test == 'kruskal_wallis_test':
        # Placeholder for performing Kruskal-Wallis Test
        return "Result of Kruskal-Wallis Test"
    elif test == 'nested_anova':
        # Placeholder for performing Nested ANOVA
        return "Result of Nested ANOVA"
    elif test == 'two_way_anova':
        # Placeholder for performing Two-Way ANOVA
        return "Result of Two-Way ANOVA"
    elif test == 'paired_t_test':
        # Placeholder for performing Paired t-Test
        return "Result of Paired t-Test"
    elif test == 'wilcoxon_signed_rank_test':
        # Placeholder for performing Wilcoxon Signed-Rank Test
        return "Result of Wilcoxon Signed-Rank Test"

def perform_multiple_measurement_test(test):
    if test == 'linear_regression_and_correlation':
        # Placeholder for performing Linear Regression and Correlation
        return "Result of Linear Regression and Correlation"
    elif test == 'spearman_rank_correlation':
        # Placeholder for performing Spearman Rank Correlation
        return "Result of Spearman Rank Correlation"
    elif test == 'polynomial_regression':
        # Placeholder for performing Polynomial Regression
        return "Result of Polynomial Regression"
    elif test == 'analysis_of_covariance':
        # Placeholder for performing Analysis of Covariance
        return "Result of Analysis of Covariance"
    elif test == 'multiple_regression':
        # Placeholder for performing Multiple Regression
        return "Result of Multiple Regression"
    elif test == 'simple_logistic_regression':
        # Placeholder for performing Simple Logistic Regression
        return "Result of Simple Logistic Regression"
    elif test == 'multiple_logistic_regression':
        # Placeholder for performing Multiple Logistic Regression
        return "Result of Multiple Logistic Regression"

@bp.route('/normalize', methods=['POST','GET'])
def normalize():
    transformed_df = df.copy()
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
    sns.scatterplot(x=transformed_df['X'], y=transformed_df['Y'])
    graph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_norm'+'.png'
    plt.savefig(os.path.join(current_app.root_path, 'static/'+ graph))          
    return render_template('display_excel.html', filename=file.filename, graph=graph,tables=[df.to_html(classes='data')], titles=df.columns.values, transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))

@bp.route('/transform', methods=['POST','GET'])
def transform():
    transformed_df= df.copy()
    biochem_transform = request.form.get('biochem')
    function = request.form.get('function')
    hill_input = request.form.get('hill_input')
    user_input = request.form.get('user_input')
    if biochem_transform == "eadie-hofstee":
        transformed_df['X'] = df['Y'] / df['X']
    elif biochem_transform == "hanes-woolf":
        transformed_df['Y'] = df['X'] / df['Y']
    elif biochem_transform == "hill":
        transformed_df['X'] = np.log10(df['X'])
        transformed_df['Y'] = np.log10(df['Y'] / (float(hill_input) - df['Y']))
    elif biochem_transform == "log-log":
        transformed_df['X'] = np.log10(df['X'])
        transformed_df['Y'] = np.log10(df['Y'])
    elif biochem_transform == "scatchard":
        transformed_df['Y'] = df['Y'] / df['X']
        transformed_df['X'] = df['Y']
    if function == "y*k":
        transformed_df['Y']=df['Y']*float(user_input)
    elif function == "y+k":
        transformed_df['Y']=df['Y']+float(user_input)
    elif function == "y-k":
        transformed_df['Y']=df['Y']-float(user_input)
    elif function == "y/k":
        transformed_df['Y']=df['Y']/float(user_input)
    elif function == "ysquared":
        transformed_df['Y']=df['Y']^2
    elif function == "y^k":
        transformed_df['Y']=df['Y']^float(user_input)
    elif function == "logy":
        transformed_df['Y']=np.log10(df['Y'])
    elif function == "-logy":
        transformed_df['Y']=-1*np.log10(df['Y'])
    elif function == "lny":
        transformed_df['Y']=np.log(df['Y'])
    elif function == "10^y":
        transformed_df['Y']=10^df['Y']
    elif function == "e^y":
        transformed_df['Y']=math.exp(df['Y'])
    elif function == "1/y":
        transformed_df['Y']=1/df['Y']
    elif function == "sqrty":
        transformed_df['Y']=math.sqrt(df['Y'])
    elif function == "logity":
        transformed_df['Y']=logit(df['Y'])
    elif function == "zscorey":
        transformed_df['Y']=zscore(df['Y'])
    elif function == "siny":
        transformed_df['Y']=math.sin(df['Y'])
    elif function == "cosy":
        transformed_df['Y']=math.cos(df['Y'])
    elif function == "tany":
        transformed_df['Y']=math.tan(df['Y'])
    elif function == "arcsiny":
        transformed_df['Y']=math.asin(df['Y'])
    elif function == "absy":
        transformed_df['Y']=abs(df['Y'])
    elif function == "x/y":
        transformed_df['Y']=df['X']/df['Y']
    elif function == "y/x":
        transformed_df['Y']=df['Y']/df['X']
    elif function == "y-x":
        transformed_df['Y']=df['Y']-df['X']
    elif function == "y+x":
        transformed_df['Y']=df['Y']+df['X']
    elif function == "y*x":
        transformed_df['Y']=df['Y']*df['X']
    elif function == "x-y":
        transformed_df['Y']=df['X']-df['Y']
    elif function == "k-y":
        transformed_df['Y']=float(user_input)-df['Y']
    elif function == "k/y":
        transformed_df['Y']=float(user_input)/df['Y']
    elif function == "log2y":
        transformed_df['Y']=np.log2(df['Y'])
    elif function == "2^y":
        transformed_df['Y']=2^df['Y']
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=transformed_df['X'], y=transformed_df['Y'])
    graph=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_trans'+'.png'
    plt.savefig(os.path.join(current_app.root_path, 'static/'+ graph))          
    return render_template('display_excel.html', filename=file.filename, graph=graph,tables=[df.to_html(classes='data')], titles=df.columns.values, transformed_df=(transformed_df.to_dict(orient='records') if df is not None else None))

def transform_concentrations(df):
    # Placeholder for transformation
    return df
def area_under_curve(df):
    # Placeholder for transformation
    return df
def fraction_of_total(df):
    # Placeholder for transformation
    return df
def transpose(df):
    # Placeholder for transformation
    return df

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


