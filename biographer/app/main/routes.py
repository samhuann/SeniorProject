from datetime import datetime, timezone
from flask import render_template, flash, redirect, url_for, request, g, \
    current_app
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
from wtforms.validators import ValidationError
import os


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
        return 'No selected file'
    if file:
        # Save the file to a temporary location
        file_path = os.path.join(current_app.root_path, 'uploads/'+ file.filename)
        file.save(file_path)
        
        # Read the Excel file
        global df
        df = pd.read_excel(file_path)
        # Render the template with the Excel data
       
        return render_template('display_excel.html', filename=file.filename, tables=[df.to_html(classes='data')], titles=df.columns.values)
    
@bp.route('/process_option', methods=['POST'])
def process_option():
    regression_type = request.form['regression_type']
    if regression_type == 'linear':
        X = df['X']
        Y = df['Y']
        # Perform simple linear regression
        slope, intercept, r_value, p_value, std_err = linregress(X, Y)
        # Print regression parameters
        print("Slope:", slope)
        print("Intercept:", intercept)
        print("R-squared:", r_value**2)
        print("P-value:", p_value)
        print("Standard error:", std_err)
        # Plot the data and regression line
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X, y=Y, label='Data')
        sns.lineplot(x=X, y=slope*X + intercept, color='red', label='Linear Regression')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Simple Linear Regression")
        figname=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'_linr'+'.png'
        plt.legend()
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ figname))
    elif regression_type == 'logistic':
        # Logistic function
        def logistic_function(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))
        X = df['X']
        Y = df['Y']

            # Perform logistic regression
        popt, pcov = curve_fit(logistic_function, X, Y)

            # Extract parameters
        L, k, x0 = popt

            # Print regression parameters
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
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Simple Logistic Regression")
        figname=datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")+'.png'
        plt.legend()
        plt.savefig(os.path.join(current_app.root_path, 'static/'+ figname))
    print(df.head())
    return render_template('display_excel.html', filename=file.filename, figname=figname, tables=[df.to_html(classes='data')], titles=df.columns.values)
         



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


