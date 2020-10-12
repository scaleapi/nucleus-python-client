import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements_links(fi: str):
    _e = "-e "

    def proc_req(r):
        r = r.strip()
        if len(r) == 0 or any(map(lambda x: r.startswith(x), ["#", "."])):
            return None
        if r.startswith(_e):
            r = r[r.rindex("=") + 1 :]
        return r

    def proc_link(r):
        r = r.strip()
        if len(r) == 0 or not r.startswith(_e):
            return None
        return r[len(_e) :]

    with open(fi, "rt") as rt:
        lines = rt.read().splitlines()
        reqs = list(filter(None, map(proc_req, lines)))
        links = list(filter(None, map(proc_link, lines)))
    return reqs, links


requires, links = read_requirements_links("requirements.txt")

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
