import setuptools
import sys

if sys.version_info < (3, 5):
    sys.exit("Sorry, Python < 3.5 is not supported")

with open("README.md", "r") as fh:
    long_description = fh.read()


print(links)

setuptools.setup(
    name="nucleus",
    version="0.0.1",
    author="Scale AI",
    author_email="support@scale.com",
    description="The official Python client library for Nucleus, the Data Platform for AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scaleapi/nucleus-python-client",
    packages=setuptools.find_packages(),
    install_requires=requires,
    dependency_links=links,
    python_requires=">=3.6",
)
