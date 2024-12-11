## TensorFlow Setup
* Reference: [This URL](https://www.tensorflow.org/install/pip?hl=en#linux)
* To install tensorflow with GPU enabled, take the following steps:
	* First, install tensorflow with pip:
	```bash
	python3 -m pip install 'tensorflow[and-cuda]'
	```
	* NOTE: the package's name **IS EXACTLY** "tensorflow[and-cuda]"!
* TroubleShooting:
	* If you run across the error:
	```bash
	ERROR: Ignored the following versions that require a different python version: 2.1.0 Requires-Python >=3.10; 2.1.0rc1 Requires-Python >=3.10; 2.1.1 Requires-Python >=3.10; 2.1.2 Requires-Python >=3.10; 2.1.3 Requires-Python >=3.10; 2.2.0 Requires-Python >=3.10; 2.2.0rc1 Requires-Python >=3.10
	ERROR: Could not find a version that satisfies the requirement tensorrt-libs==8.6.1; extra == "and-cuda" (from tensorflow[and-cuda]) (from versions: 9.0.0.post11.dev1, 9.0.0.post12.dev1, 9.0.1.post11.dev4, 9.0.1.post12.dev4, 9.1.0.post11.dev4, 9.1.0.post12.dev4, 9.2.0.post11.dev5, 9.2.0.post12.dev5, 9.3.0.post11.dev1, 9.3.0.post12.dev1)
	ERROR: No matching distribution found for tensorrt-libs==8.6.1; extra == "and-cuda"
	
	```
	It's due to the pip cannot find the proper version of the tensorrt. Try mannual install it first:
	
	```bash
	pip install https://pypi.nvidia.com/tensorrt-libs/tensorrt_libs-8.6.1-py2.py3-none-manylinux_2_17_x86_64.whl	
	```
	
	Note: the relevant pacages are tensorflow(2.15.0), cuda(12.2), tensorrt(8.6.1)
