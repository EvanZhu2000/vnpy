import pandas as pd
from vnpy_self.pt_tools import *
import sys
import paramiko
from io import StringIO

# Wrote by POE
def connect_with_ssh(remote_file_path, hostname, username, password):
  port = 22

  ssh = paramiko.SSHClient()
  ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
  ssh.connect(hostname, port, username, password)

  sftp = ssh.open_sftp()

  try:  
    with sftp.open(remote_file_path, 'r') as file:
        df = pd.read_csv(file)
        return df
  except IOError as e:
      print(f"IOError: {e}")
  except FileNotFoundError:
      print(f"File not found: {remote_file_path}")
  except Exception as e:
      print(f"An unexpected error occurred: {e}")
      
def show(fig):
    import io
    import plotly.io as pio
    from PIL import Image
    buf = io.BytesIO()
    pio.write_image(fig, buf)
    img = Image.open(buf)
    img.show() 

def run(hostname, username, password, file_path):
    df = connect_with_ssh(file_path, hostname, username, password)
    d = df.set_index('Unnamed: 0').dropna(how='all')
    d.index = pd.to_datetime(d.index)
    print(d.tail())
    show(plot(1000000, d.sum(axis=1)))

if __name__ == "__main__":
    # hostname = sys.argv[1]
    # username = sys.argv[2]
    # password = sys.argv[3]
    # file_path = sys.argv[4]
    # run(hostname, username, password, file_path)
    
    ### Example
    run('192.168.91.124', 'uat', 'evan@cash1q2', '/home/uat/miniconda3/envs/vnpy3/lib/python3.10/site-packages/vnpy_self/analysis/data/strategy2_expected_pnl.csv')
