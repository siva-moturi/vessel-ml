import setuptools


setuptools.setup(
    name="vesselml",
    version="0.0.9",
    author="Siva Moturi",
    author_email="siva.moturi@pfizer.com",
    description="Vessel ML library for cloud based ML training and model deployment",
    url="http://bitbucket-insightsnow.pfizer.com:7990/scm/ga/vessel-ml.git",
    packages=setuptools.find_packages(),
    install_requires=[
              'pandas',
              'kubernetes',
              'kfmd',
              'fairing',
              'google-cloud-storage',
              'joblib',
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
