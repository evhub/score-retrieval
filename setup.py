import setuptools


setuptools.setup(
    name="score_retrieval",
    version=0.1,
    install_requires=[
        "numpy",
        "scipy",
        "pdf2image",
    ],
    packages=setuptools.find_packages(),
)
