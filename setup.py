from setuptools import setup, find_packages

setup(
    name="muscatreader",
    version="0.0.1",
    description=".dat to .nc file conversion",
    url="http://github.com/jpdeleon/muscatreader",
    author="Jerome de Leon",
    author_email="jpdeleon.bsap@gmail.com",
    license="MIT",
    packages=None,  # or find_packages(),
    #include_package_data=True,
    #scripts=[None],
    zip_safe=False,
    install_requires=[
        "astropy",
        "xarray",
        "pandas",
        "tqdm",
    ],
)
