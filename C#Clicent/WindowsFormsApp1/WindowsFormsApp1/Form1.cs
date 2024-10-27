using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using OpenCvSharp;
using MaterialSkin;
using MaterialSkin.Controls; // 引入 MaterialSkin 控件
using System.Threading.Tasks;
using Guna.UI2.WinForms;

namespace WindowsFormsApp1
{
    public partial class MainForm :  MaterialForm
    {
        private VideoCapture _capture;
        private InferenceSession _session;
        private PictureBox _pictureBox;
        private Guna2Button _startButton, _sButton;
        private Guna2HtmlLabel _resultBox;
        private Timer _timer;
        private string[] classes = { "镜铁矿", "钛铁矿", "方铅矿", "斑铜矿", "闪锌矿", "磁铁矿", "锰矿", "菱镁矿", "黄铁矿", "赤铁矿", "铝土矿", "硬锰矿", "蓝铜矿", "辰砂", "黄铜矿" };
        //private AxWindowsMediaPlayer _mediaPlayer; // 用于播放视频
        private AxWMPLib.AxWindowsMediaPlayer _mediaPlayer;

        public MainForm()
        {
            InitializeComponent();
            // 设置应用程序名称
            this.Text = "矿石识别";

            // 设置应用程序图标
            this.Icon = new Icon("appicon.ico"); // 替换为你的图标路径


            this.ClientSize = new System.Drawing.Size(800, 600);
            // 设置窗体边框样式
            this.FormBorderStyle = FormBorderStyle.Sizable; // 设置为正常的可调整大小的窗体
            // 创建媒体播放器控件
            _mediaPlayer = new AxWMPLib.AxWindowsMediaPlayer
            {
                Dock = DockStyle.Top,
                Height = 400,
                Visible = false
            };

            // 创建和设置 PictureBox
            _pictureBox = new PictureBox
            {
                Dock = DockStyle.Top,
                Height = 400,
                SizeMode = PictureBoxSizeMode.Zoom,
                BorderStyle = BorderStyle.FixedSingle
            };

            // 创建和设置开始识别按钮
            _startButton = new Guna2Button
            {
                Text = "开始识别",
                Dock = DockStyle.Top,
                Height = 50,
                FillColor = Color.FromArgb(30, 144, 255), // 使用柔和的蓝色
                ForeColor = Color.White, // 字体颜色为白色
                BorderColor = Color.Transparent, // 去掉边框
                BorderThickness = 0
            };

            // 创建和设置结果文本框
            _resultBox = new Guna2HtmlLabel
            {
                Dock = DockStyle.Fill,
                Font = new Font("Arial", 12),
                BackColor = Color.WhiteSmoke,
                ForeColor = Color.Black
            };

            // 创建选择图片按钮
            _sButton = new Guna2Button
            {
                Text = "选择图片",
                Dock = DockStyle.Top,
                Height = 50,
                FillColor = Color.FromArgb(70, 130, 180), // 使用更深的蓝色
                ForeColor = Color.White, // 字体颜色为白色
                BorderColor = Color.Transparent, // 去掉边框
                BorderThickness = 0
            };


            // 添加控件到窗体
            Controls.Add(_resultBox);
            Controls.Add(_startButton);
            Controls.Add(_pictureBox);
            Controls.Add(_mediaPlayer);
            Controls.Add(_sButton);
            // 绑定事件
            _startButton.Click += StartButton_Click;
            _sButton.Click += LoadImage_Click;

            // 尝试加载 ONNX 模型
            try
            {
                _session = new InferenceSession("model6.onnx");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"加载模型失败: {ex.Message}", "错误", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            // 初始化摄像头捕获
            _capture = new VideoCapture(0);
            _timer = new Timer { Interval = 30 };
            _timer.Tick += Timer_Tick;
            _timer.Start();
        }

        private void Timer_Tick(object sender, EventArgs e)
        {
            using (var frame = new OpenCvSharp.Mat())
            {
                if (_capture.Read(frame)) // 从摄像头读取帧
                {
                    // 将 Mat 转换为 Bitmap
                    _pictureBox.Image?.Dispose(); // 释放之前的图像
                    _pictureBox.Image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame); // 设置新图像
                }
            }
        }

        private void LoadImage_Click(object sender, EventArgs e)
        {
            // 选择要加载的图片文件
            using (var openFileDialog = new OpenFileDialog())
            {
                openFileDialog.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp";

                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    // 加载图片
                    var selectedBitmap = new Bitmap(openFileDialog.FileName);
                    _pictureBox.Image?.Dispose(); // 释放之前的图像
                    _pictureBox.Image = selectedBitmap; // 设置新图像

                    // 进行模型预测
                    string result = Predict(selectedBitmap);
                    _resultBox.Text = result; // 显示预测结果
                }
            }
        }

        private void StartButton_Click(object sender, EventArgs e)
        {
            if (_startButton.Text == "开始识别")
            {
                var bitmap = new Bitmap(_pictureBox.Image);
                string result = Predict(bitmap);
                _resultBox.Text = result;

                _startButton.Text = "重新识别";
                _pictureBox.Visible = false;
            }
            else
            {
                _mediaPlayer.URL = ""; // 停止播放
                _mediaPlayer.Visible = false;
                _pictureBox.Visible = true;
                _resultBox.Text = ""; // 显示预测结果

                _pictureBox.Invalidate(); // 标记需要重绘
                _mediaPlayer.Invalidate(); // 标记媒体播放器需要重绘
                // 确保界面刷新
                this.Invalidate(); // 强制界面重绘
                this.Refresh(); // 刷新界面
                _startButton.Text = "开始识别";
            }
 
        }

        private string Predict(Bitmap bitmap)
        {
            // 在将图像转换为 Tensor 之前调整大小
            bitmap = ResizeBitmap(bitmap, 224, 224);
            // 将图像转换为 Tensor
            var tensor = BitmapToTensor(bitmap);

            var inputName = _session.InputMetadata.First().Key; // 获取第一个输入的名称
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };

            using (var results = _session.Run(inputs))
            {
                var output = results.First().AsEnumerable<float>().ToArray();
                return ProcessOutput(output); // 返回对应的类别
            }
        }

        public Bitmap ResizeBitmap(Bitmap bitmap, int width, int height)
        {
            // 创建一个新的 Bitmap 对象，用于保存调整大小后的图像
            var resizedBitmap = new Bitmap(width, height);

            // 使用 Graphics 对象绘制调整大小后的图像
            using (var graphics = Graphics.FromImage(resizedBitmap))
            {
                graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                graphics.DrawImage(bitmap, 0, 0, width, height);
            }

            return resizedBitmap;
        }

        private Tensor<float> BitmapToTensor(Bitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;
            var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var color = bitmap.GetPixel(x, y);
                    tensor[0, 0, y, x] = color.R / 255.0f; 
                    tensor[0, 1, y, x] = color.G / 255.0f; 
                    tensor[0, 2, y, x] = color.B / 255.0f;
                }
            }
            return tensor;
        }

        private string ProcessOutput(float[] output)
        {
            int predictedIndex = Array.IndexOf(output, output.Max());
            string className = classes[predictedIndex];
            _mediaPlayer.Visible = true; // 显示播放器
            PlayVideoForClass(className); // 播放对应类别的视频
            
            return $"识别结果: {classes[predictedIndex]}"; 
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            _capture?.Dispose();
            _session?.Dispose();
            base.OnFormClosing(e);
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            // 这里可以进行窗体加载时的设置
        }


        private void PlayVideoForClass(string className)
        {
            string videoPath = GetVideoPathForClass(className); // 获取视频路径
            Console.WriteLine($"播放视频: {videoPath}");

            if (!string.IsNullOrEmpty(videoPath))
            {
                // 隐藏图片框
                _pictureBox.Visible = false;

                // 确保界面刷新
                this.Invalidate(); // 强制界面重绘
                this.Refresh(); // 刷新界面

                _mediaPlayer.uiMode = "none"; // 隐藏播放按钮和其他 UI 元素
                _mediaPlayer.URL = videoPath; // 设置媒体播放器的 URL
                _mediaPlayer.Ctlcontrols.play(); // 播放视频
            }
        }

        private string GetVideoPathForClass(string className)
        {
                // 视频映射字典
            var videoMapping = new Dictionary<string, string>
            {
                { "镜铁矿", "videos/1.mp4" },
                { "菱镁矿", "videos/2.mp4" },
                { "黄铁矿", "videos/3.mp4" },
                { "赤铁矿", "videos/4.mp4" },
                { "铝土矿", "videos/5.mp4" },
                { "硬锰矿", "videos/6.mp4" },
                { "蓝铜矿", "videos/7.mp4" },
                { "辰砂", "videos/8.mp4" },
                { "黄铜矿", "videos/9.mp4" },
                { "钛铁矿", "videos/10.mp4" },
                { "方铅矿", "videos/11.mp4" },
                { "斑铜矿", "videos/12.mp4" },
                { "闪锌矿", "videos/13.mp4" },
                { "磁铁矿", "videos/14.mp4" },
                { "锰矿", "videos/15.mp4" }
            };

            // 根据类别名称返回相应的视频文件路径
            if (videoMapping.TryGetValue(className, out string videoPath))
            {
                return videoPath; // 返回匹配到的视频路径
            }

            return null; // 未匹配到的类别返回 null
        }
    }
}
