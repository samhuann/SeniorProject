from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
from datetime import datetime, timezone
from flask import render_template, flash, redirect, url_for, request, g, \
    current_app, send_from_directory
from flask_login import current_user, login_required
import sqlalchemy as sa
from app import db
from app.main.forms import EditProfileForm
from app.models import User, Post
from app.main import bp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress 
import pandas as pd
from scipy.optimize import curve_fit
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame, Series  # for convenience
import pims
import trackpy as tp

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
        file_path = os.path.join(current_app.root_path, 'uploads/'+ file.filename)
        file.save(file_path)
        
        # Read the Excel file
        global df
        df = pd.read_excel(file_path)
        # Render the template with the Excel data
       
        return render_template('display_excel.html', filename=file.filename, tables=[df.to_html(classes='data')], titles=df.columns.values)


@bp.route('/analyze', methods=['POST'])
def analyze_excel():
    file_path = request.form['file_path']
    transformation = request.form['transformation']
    test_nominal = request.form['test_nominal']
    test_one_measurement = request.form['test_one_measurement']
    test_multiple_measurement = request.form['test_multiple_measurement']

    # Perform transformation or statistical tests
    df = pd.read_excel(file_path)
    transformed_df = None

    # Perform transformation based on selected option
    if transformation == 'normalization':
        # Apply normalization
        transformed_df = normalize_data(df)
    elif transformation == 'transformation':
        # Apply transformation
        transformed_df = transform_data(df)
    elif transformation == 'pruning':
        # Apply pruning
        transformed_df = prune_data(df)

    # Perform statistical tests based on selected options
    test_results = {}
    if test_nominal == 'chi_square':
        test_results['chi_square'] = perform_chi_square_test(transformed_df)
    elif test_nominal == 'fisher_exact':
        test_results['fisher_exact'] = perform_fishers_exact_test(transformed_df)

    if test_one_measurement == 't_test':
        test_results['t_test'] = perform_t_test(transformed_df)
    elif test_one_measurement == 'anova':
        test_results['anova'] = perform_anova(transformed_df)

    if test_multiple_measurement == 'correlation':
        test_results['correlation'] = perform_correlation_test(transformed_df)
    elif test_multiple_measurement == 'regression':
        test_results['regression'] = perform_regression_test(transformed_df)

    # Render the template with the updated Excel data, dropdowns, and test results
    return render_template('display_excel.html', file_path=file_path, test_results=test_results)


# Placeholder functions for transformation and statistical tests
def normalize_data(df):
    # Placeholder for normalization
    return df

def transform_data(df):
    # Placeholder for transformation
    return df

def prune_data(df):
    # Placeholder for pruning
    return df

def perform_chi_square_test(df):
    # Placeholder for performing Chi-Square test
    return "Chi-Square Test Result"

def perform_fishers_exact_test(df):
    # Placeholder for performing Fisher's Exact test
    return "Fisher's Exact Test Result"

def perform_t_test(df):
    # Placeholder for performing T-Test
    return "T-Test Result"

def perform_anova(df):
    # Placeholder for performing ANOVA
    return "ANOVA Result"

def perform_correlation_test(df):
    # Placeholder for performing correlation test
    return "Correlation Test Result"

def perform_regression_test(df):
    # Placeholder for performing regression test
    return "Regression Test Result"












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


