from setuptools import setup, find_packages

setup(
    name="rice-disease-classifier",
    version="0.1.0",
    description="MLOps project for rice leaf disease classification",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "isort>=5.13.2",
        ]
    },
    entry_points={
        "console_scripts": [
            "rice-train=src.train:main",
            "rice-predict=src.predict:main",
        ]
    },
)
