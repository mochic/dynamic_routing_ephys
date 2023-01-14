from setuptools import find_packages, setup

packages = find_packages()

setup(name="sync",
      version="1.4.0",
      description="AIBS Sync Package",
      author="derricw",
      author_email="derricw@alleninstitute.org",
      url="http://stash.corp.alleninstitute.org/projects/ENG/repos/sync/browse",
      packages=packages,
      install_requires=[],
      include_package_data=True,
      package_data={
          "": ['*.png', '*.ico', '*.jpg', '*.jpeg'],
          },
      entry_points={
          'console_scripts': [
              'sync_gui = sync.gui.sync_gui:main',
              'sync_device = sync.zro.sync_device:main',
          ],
      }
      )
