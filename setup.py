from setuptools import setup, find_packages

setup(
    name='quantedge',
    version='0.0.1',
    description='QuantEdge trading analysis package',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    install_requires=[],
)
