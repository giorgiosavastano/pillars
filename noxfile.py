import nox


@nox.session
def tests(session):
    session.install("pip", "numpy", "pytest", "pytest-benchmark" "scipy")
    session.run("pip", "install", ".", "-v")
    session.run("pytest")
