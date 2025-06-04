
#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = ["gunicorn", "werkzeug", "flask"]

setup(
    author="Arne Gross",
    author_email='arne.gross@imtek.uni-freiburg.de',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Project contains code for local control and cloud coordination in GrECCo",
    install_requires=requirements,
    include_package_data=True,
    name='grecco_sim',
    packages=find_packages(include=['grecco_sim', 'grecco_sim.*']),
    package_data={'grecco_sim': ['data/heat_pump_database/heat_pump_database_short_version.csv'],},
    data_files=[
        ('data/heat_pump_database', ['data/heat_pump_database/heat_pump_database_short_version.csv'])
    ],
    url='https://github.com/ArneJGross/grecco_sim',
    version='0.1.0',
    zip_safe=False,
)