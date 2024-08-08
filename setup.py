from setuptools import setup, find_packages

setup(
    name='advanced-time-series-analysis',
    version='0.1.0',
    description='Advanced time-series analysis and forecasting library',
    author='shinoj cm',
    author_email='shinojcm01@gmail.com',
    url='https://github.com/fridayowl/AdvancedTimeSeriesAnalysis',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'tensorflow',
        'statsmodels',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)