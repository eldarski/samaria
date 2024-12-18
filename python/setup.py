from setuptools import setup, find_packages

setup(
    name="samaria_contrast",
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    package_data={
        "samaria_contrast": ["py.typed", "*.pyi"],
    },
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.8.0",
        "sentencepiece>=0.1.96",
        "opencv-python>=4.5.0",
    ],
    extras_require={
        "test": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
        ],
        "dev": [
            "black>=22.3.0",
            "isort>=5.10.1",
            "flake8>=4.0.1",
            "mypy>=0.950",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Japanese text and image contrastive learning library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/<organization>/samaria_contrast",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
) 
