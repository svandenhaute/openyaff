import setuptools


setuptools.setup(
        name='openyaff',
        version='0.0.1',
        author='Sander Vandenhaute',
        packages=['openyaff'],
        python_requires='>=3.6',
        entry_points={
            'console_scripts': [
                'openyaff = openyaff.cli:main',
                ]
            }
        )
