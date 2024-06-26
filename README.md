# FinalCase-MiPower
The final project for the Patika.dev &amp; Mİ-POWER: Women Empowerment Program.
# SalaryPrediction Project -Nano Grup

This is a basic Flask application Dockerized for easy deployment. 
## If you would like to visit our website :

[Our Website](https://salaryprediction-56qr.onrender.com/) 

## Final version of the project 

https://github.com/Nanogrup/SalaryPrediction/assets/84244385/6da5e577-3aaf-4f92-bd43-d67cb5b6ce21

## Project Overview 
### Home Page
![Ekran Resmi 2024-06-02 01 13 07](https://github.com/Nanogrup/SalaryPrediction/assets/103145955/43cda262-196c-444f-a65b-20ac83d12a2b)

### Data Analysis Page

<img width="1052" alt="Screenshot 2024-06-22 2029421" src="https://github.com/gamzebozkus/FinalCase-MiPower/assets/84244385/04512026-c179-4ca1-a0ca-abc10eec8a6a">

![Screenshot 2024-06-22 203053](https://github.com/gamzebozkus/FinalCase-MiPower/assets/84244385/d8c7291f-0a2e-4bba-9120-b6933dcfa44e)

![Screenshot 2024-06-22 203105](https://github.com/gamzebozkus/FinalCase-MiPower/assets/84244385/4c16c5cd-85e3-4449-9330-02188d31eb05)

![Screenshot 2024-06-22 203120](https://github.com/gamzebozkus/FinalCase-MiPower/assets/84244385/1d9a5b3d-0646-448f-8902-a9b1d8968b28)

![Screenshot 2024-06-22 203134](https://github.com/gamzebozkus/FinalCase-MiPower/assets/84244385/a998c7c5-24d1-4c2d-86c7-7861eaa3aa0a)

### Salary Prediction Page
![Ekran Resmi 2024-06-02 01 14 03](https://github.com/Nanogrup/SalaryPrediction/assets/103145955/8ed339fb-4bc0-4f17-be88-8d46f16c6ade)
![Ekran Resmi 2024-06-02 01 14 09](https://github.com/Nanogrup/SalaryPrediction/assets/103145955/d5c07f05-976e-4a88-9736-375ae8741b85)

### Result Page
![Ekran Resmi 2024-06-02 01 14 29](https://github.com/Nanogrup/SalaryPrediction/assets/103145955/d17c140b-f767-45e1-8aef-97cc801b0fce)
![Ekran Resmi 2024-06-02 01 14 42](https://github.com/Nanogrup/SalaryPrediction/assets/103145955/0b3f894d-b9c8-4803-bf6a-d2a45d1f7d89)


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Nanogrup/SalaryPrediction.git
cd SalaryPrediction/flask_app
```
## Without Docker :
### 2. Install python 

### 3. Install dependencies(flask,...)
* You can download all the necessary libraries from the requirements.txt file with the command below:
```bash
pip install -r requirements.txt
```
* Or you can download each library individually with pip install as below:
```bash
pip install Flask
pip install ...
```

### 4. Start the flask app:
```bash
python app.py
```

### 5. Open a web browser and go to:
```
http://127.0.0.1:5000/
```

## With Docker : 
### 2. Install Docker

* Download and install Docker Desktop from the [Docker website](https://www.docker.com/products/docker-desktop).


### 3. Build the Docker image:
```bash
docker build -t flask-app .
```

### 4. Run the Docker container:
```bash
docker run -p 5000:5000 flask-app
```

### 5. Open a web browser and go to:
```
http://127.0.0.1:5000/
```

## With Virtual Environment(venv): 
### 2. Install Python
### 3. Create venv directory
```bash
python3 -m virtual environmentpath/to/venv
```
### 4. Activate virtual environment
```bash
source path/to/venv/bin/activate
```
### 5. Install dependencies(flask,...)
### 6. Start the flask app
### 7. Open a web browser and go to url

## Files and Directories

- `app.py`: Main Flask application file.
- `requirements.txt`: List of required Python packages.
- `Dockerfile`: Docker configuration file.
- `templates/`: Directory containing HTML templates.
- `static/`: Directory containing static files like CSS.
- `data/`: Directory containing csv file and jupyter notebooks.
- `machine_learning.py`: Model training application file.
- `salary_model.pkl`: Model file.
```csharp
flask_app/
    ├── app.py
    ├── machine_learning.py
    ├── requirements.txt
    ├── Dockerfile
    ├── salary_model.pkl
    ├── templates/
    │   ├── base.html
    │   ├── index.html
    │   ├── salary_prediction.html
    │   └── data_analysis.html
    │   └── result.html
    ├── static/
    │    └── style.css
    ├── data/
    │   ├── Salary Analysis.ipynb
    │   └── salaryData.csv
```
