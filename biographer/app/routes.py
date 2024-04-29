from urllib.parse import urlsplit
from flask import render_template, flash, redirect, url_for, request
from flask_login import login_user, logout_user, current_user, login_required
import sqlalchemy as sa
from app import app, db
from app.forms import EditProfileForm, LoginForm, RegistrationForm, ResetPasswordForm, ResetPasswordRequestForm
from app.models import User
import pandas as pd
from datetime import datetime, timezone
from app.email import send_password_reset_email
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress 
from scipy.optimize import curve_fit

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    user = User.verify_reset_password_token(token)
    if not user:
        return redirect(url_for('index'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        flash('Your password has been reset.')
        return redirect(url_for('login'))
    return render_template('reset_password.html', form=form)

@app.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = ResetPasswordRequestForm()
    if form.validate_on_submit():
        user = db.session.scalar(
            sa.select(User).where(User.email == form.email.data))
        if user:
            send_password_reset_email(user)
        flash('Check your email for the instructions to reset your password')
        return redirect(url_for('login'))
    return render_template('reset_password_request.html',
                           title='Reset Password', form=form)



@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash('Your changes have been saved.')
        return redirect(url_for('edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title='Edit Profile',
                           form=form)

@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.now(timezone.utc)
        db.session.commit()

@app.route('/')
@app.route('/index')
@login_required

def index():
    return render_template('index.html', title='Home')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = db.session.scalar(
            sa.select(User).where(User.username == form.username.data))
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or urlsplit(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/index', methods=['GET', 'POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:        
        file.save(file.filename)
        # Read the Excel file into a DataFrame
        data = pd.read_excel(file.filename)
        regression_type = request.form['regression_type']
        if regression_type == 'linear':
            X = data['X']
            Y = data['Y']
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
            plt.legend()
            plt.show()            
        elif regression_type == 'logistic':
            # Logistic function
            def logistic_function(x, L, k, x0):
                return L / (1 + np.exp(-k * (x - x0)))
            X = data['X']
            Y = data['Y']

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
            plt.legend()
            plt.show()
        print(data.head())
        return render_template('index.html', filename=file.filename, tables=[data.to_html(classes='data')], titles=data.columns.values)
    
@app.route('/user/<username>')
@login_required
def user(username):
    user = db.first_or_404(sa.select(User).where(User.username == username))
    return render_template('user.html', user=user)

