from distutils.core import setup
setup(
  name = 'paragrid',         # How you named your package folder (MyLib)
  packages = ['paragrid'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Simple parallelized grid search using scikits Optimizer to find the best hyperparameters',   # Give a short description about your library
  author = 'Malte Algren',                   # Type in your name
  author_email = 'malte__algren@hotmail.com',      # Type in your E-Mail
  url = 'https://github.com/malteal/paragrid',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/malteal/paragrid/archive/0.0.1.tar.gz',    # I explain this later on
  keywords = ['machinelearning', 'gridsearchcv', 'parallel-computing'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'scikit-optimize',
          'sklearn',
          'matplotlib',
          'xgboost',
          'lightgbm',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)