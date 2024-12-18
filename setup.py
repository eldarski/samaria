from setuptools import setup, find_packages

setup(
    name="samaria_contrast",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.8.0",
        "sentencepiece>=0.1.96",
        "opencv-python>=4.5.0",
    ],
    extras_require={
        "test": ["pytest>=6.0.0", "opencv-python>=4.0.0"],
        "dev": [
            "black",
            "isort",
            "flake8",
            "mypy",
            "pytest-cov",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Japanese text and image contrastive learning library",
    long_description=open("python/README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
)
