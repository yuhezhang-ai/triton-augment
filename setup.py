"""
Setup configuration for triton-augment package.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
def read_file(filename):
    """Read file contents."""
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()


# Read version from __init__.py
def get_version():
    """Extract version from __init__.py."""
    version_file = os.path.join('triton_augment', '__init__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '0.1.0'


setup(
    name='triton-augment',
    version=get_version(),
    author='yuhezhang-ai',
    author_email='',
    description='GPU-accelerated image augmentation with kernel fusion using Triton',
    long_description=read_file('README.md') if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/yuhezhang-ai/triton-augment',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'triton>=2.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-benchmark>=4.0.0',
            'torchvision>=0.15.0',
            'pillow>=9.0.0',
            'matplotlib>=3.5.0',
        ],
        'examples': [
            'torchvision>=0.15.0',
            'pillow>=9.0.0',
            'matplotlib>=3.5.0',
        ],
    },
    keywords='pytorch triton gpu augmentation image-processing deep-learning',
    project_urls={
        'Bug Reports': 'https://github.com/yuhezhang-ai/triton-augment/issues',
        'Source': 'https://github.com/yuhezhang-ai/triton-augment',
    },
)

