import os
from setuptools import setup, find_packages

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Optional requirements
safety_gym_requirements = [
    "safety-gym>=0.0.1",
    "mujoco-py>=2.1.2.14"
]

metadrive_requirements = [
    "metadrive-simulator>=0.2.4"
]

dev_requirements = [
    "pytest>=6.2.5",
    "pytest-cov>=2.12.1",
    "black>=21.9b0",
    "isort>=5.9.3",
    "flake8>=3.9.2"
]

setup(
    name="srlnbc",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Implementation of Model-Free Safe Reinforcement Learning Through Neural Barrier Certificate",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/srlnbc",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/srlnbc/issues",
        "Documentation": "https://github.com/yourusername/srlnbc#readme",
        "Source Code": "https://github.com/yourusername/srlnbc",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "safety-gym": safety_gym_requirements,
        "metadrive": metadrive_requirements,
        "dev": dev_requirements,
        "all": safety_gym_requirements + metadrive_requirements + dev_requirements
    },
    entry_points={
        "console_scripts": [
            "srlnbc-train=srlnbc.scripts.train:main",
            "srlnbc-eval=srlnbc.scripts.evaluate:main",
            "srlnbc-benchmark=srlnbc.scripts.benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "srlnbc": [
            "configs/*.yaml",
            "configs/*.json",
        ]
    },
    # Add any additional data files
    data_files=[
        ("configs", ["srlnbc/configs/default_configs.py"]),
    ],
    zip_safe=False,
    # Test suite
    test_suite="tests",
)

# Post-installation messages
def post_install_message():
    """Print post-installation message."""
    print("\nThank you for installing SRLNBC!")
    print("\nTo get started, try:")
    print("  - Import the package: from srlnbc.agents import TD3SafeAgent")
    print("  - Run an example: python -m examples.classic_control")
    print("  - Check the documentation: https://github.com/yourusername/srlnbc#readme")
    print("\nFor Safety Gym environments, install additional dependencies:")
    print("  pip install srlnbc[safety-gym]")
    print("\nFor MetaDrive environments, install additional dependencies:")
    print("  pip install srlnbc[metadrive]")
    print("\nFor development tools, install:")
    print("  pip install srlnbc[dev]")
    print("\nFor all optional dependencies:")
    print("  pip install srlnbc[all]")

if __name__ == "__main__":
    setup()
    post_install_message()