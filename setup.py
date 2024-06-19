from setuptools import find_packages,setup

setup(
    name='Object detaction',
    version='0.0.1',
    author='KRSNA',
    author_email='krisnabadde@gmail.com',
    install_requires=["vidgear","ultralytics","opencv-python",'streamlit'],
    packages=find_packages()
)