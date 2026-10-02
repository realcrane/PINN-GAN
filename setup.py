from setuptools import find_packages, setup


setup(
    name="pde-gan-ntk",
    version="1.0.0",
    description="Six-PDE DEQGAN and NTK-GAN experiment package",
    packages=find_packages(),
    py_modules=["run_experiment"],
    include_package_data=True,
    package_data={"denn.config": ["*.yaml"]},
    python_requires=">=3.8,<3.9",
    install_requires=[
        "torch==2.1.2",
        "numpy==1.24.4",
        "scipy==1.10.1",
        "matplotlib==3.7.3",
        "pandas==2.0.3",
        "ipython==8.12.2",
        "PyYAML==6.0.1",
    ],
    entry_points={"console_scripts": ["pde-gan-run=run_experiment:main"]},
)
