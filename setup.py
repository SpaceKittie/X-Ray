from setuptools import setup, find_packages

# Read requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Convert exact versions (==) to minimum versions (>=)
install_requires = []
for req in requirements:
    if '==' in req:
        pkg, ver = req.split('==')
        install_requires.append(f'{pkg}>={ver}')
    else:
        install_requires.append(req)

setup(
    name="dental-xray-detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        'dev': [
            'ipykernel>=6.0.0',
            'jupyter_client>=8.0.0',
            'ipython>=8.0.0',
            'black>=23.0.0',
        ],
    },
    python_requires=">=3.7",
)
