'''
The setup.py file is an essential part of packaging and 
distributing Python projects. It is used by setuptools 
(or distutils in older Python versions) to define the configuration 
of your project, such as its metadata, dependencies, and more
'''

from setuptools import find_packages,setup
from typing import List

def get_requirements() -> List[str]:
    """
    Thiss function will return list of requirements
    
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt') as file:
            requirement_list = [req.strip() for req in file.readlines()]
            # Remove any empty lines or '-e .' if present
            requirement_list = [req for req in requirement_list if req and req != '-e .']
            
    except FileNotFoundError:
        print("Warning: requirements.txt file not found")
        
    return requirement_list

setup(
    name="Battery-Capacity-Prediction-System",
    version="0.0.1",
    author="Rahul Samant",
    author_email="rahulsamantcoc2@gmail.com",
    description="This project implements a machine learning pipeline for predicting battery capacity using various input parameters. The system includes data processing, model training, and a Flask web interface for making real-time predictions.",
    packages=find_packages(),
    install_requires=get_requirements(),
    include_package_data=True,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
)