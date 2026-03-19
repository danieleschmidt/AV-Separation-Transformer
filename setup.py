from setuptools import setup, find_packages

setup(
    name="av-separation-transformer",
    version="1.0.0",
    description="Audio-visual speech separation using transformer cross-modal fusion",
    author="Daniel Schmidt",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
    ],
)
