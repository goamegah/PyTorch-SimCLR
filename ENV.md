# ENVIRONMENT VARIABLES YOU MAY NEED TO UPLOAD CHECKPOINTS ON S3 BUCKET

### AWS credentials
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""

###  AWS S3 bucket
AWS_S3_BUCKET="bucketname"
AWS_S3_BUCKET_OBJECT_NAME="bucketname/file"

###  PROJECT DIRS
ROOT_DIR="your root directory"

To use environment variables in your code, you can follow these steps:

1. Import the `os` module in your Python script:
```python
import os
```

2. Access the values of environment variables using the `os.environ` dictionary. For example, to access the value of the `AWS_ACCESS_KEY_ID` environment variable:
```python
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
```

3. Similarly, you can access other environment variables like `AWS_SECRET_ACCESS_KEY`, `AWS_S3_BUCKET`, `AWS_S3_BUCKET_OBJECT_NAME`, and `ROOT_DIR` using the `os.environ.get()` method.

4. Make sure to set the environment variables before running your script. You can set them in your terminal or in a separate configuration file. For example, in a Linux terminal:
```bash
export AWS_ACCESS_KEY_ID="your_access_key_id"
export AWS_SECRET_ACCESS_KEY="your_secret_access_key"
export AWS_S3_BUCKET="your_bucket_name"
export AWS_S3_BUCKET_OBJECT_NAME="your_bucket_object_name"
export ROOT_DIR="your_root_directory"
```

5. Once the environment variables are set, you can access their values within your script as shown in step 2.

Remember to handle cases where the environment variables are not set or have invalid values to avoid any unexpected errors.
