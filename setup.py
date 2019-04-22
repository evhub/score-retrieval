import setuptools


setuptools.setup(
    name="score_retrieval",
    version=1.0,
    install_requires=[
        "numpy",
        "scipy",
        "pdf2image",
        "opencv-python",
        "fastdtw",
    ],
    packages=setuptools.find_packages(),
)
