from setuptools import setup, find_packages

setup(
    name='low cost robot',
    version='0.1.0',
    description='A gymnasium environment for the Koch robot',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/your_project',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'opencv-python',
        'numpy',
        'tqdm',
        # Add other dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)