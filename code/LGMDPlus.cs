/*
 * Filename: LGMDPlus.cs
 * Author: Qinbing FU
 * Date: 2019 April
 */


using System;
using System.Collections.Generic;


namespace LGMD
{
    /// <summary>
    /// Description:
    /// LGMDPlus visual neural network has a newly proposed spatial-distributed local-bias matrix affecting pre-synaptic local inhibitions.
    /// It is an extension version of the LGMDs general model with ON nad OFF pathways and adaptive inhibition mechanism.
    /// This model can realise both LGMD1 and LGMD2 neurons.
    /// a Gaussian blur is implemented on motion information spatially filtering.
    /// References: [1] Q. Fu, H. Wang, S. Yue, "Developed Visual System for Robust Collision Recognition in Various Automotive Scenes", 2019.
    ///             [2] Q. Fu, N. Bellotto, H. Wang, F. C. Rind, H. Wang, S. Yue, "A Visual Neural Network for Robust Collision Perception in Vehicle Driving Scenarios", AIAI, 2019.
    /// </summary>
    internal sealed class LGMDPlus : NewLGMDs, IComparable, IComparer<LGMDPlus>
    {
        #region LGMD FIELD

        /// <summary>
        /// a layer of gauss blurred motion information
        /// </summary>
        private float[,] blurred;
        /// <summary>
        /// gauss blur kernel
        /// </summary>
        private float[,] gauss_blur_kernel;
        /// <summary>
        /// gauss blur kernel width
        /// </summary>
        private int gauss_blur_kernel_width;
        /// <summary>
        /// local convolution kernel in the ON and OFF pathways
        /// </summary>
        private float[,] ConvK;
        /// <summary>
        /// a baseline of local inhibitory bias
        /// </summary>
        private float W_base;
        /// <summary>
        /// min value of local inhibitory bias
        /// </summary>
        private float W_min;
        /// <summary>
        /// time delays (in milliseconds) of local excitations
        /// </summary>
        private new float[] tau_E;
        /// <summary>
        /// delay coefficients of local excitations
        /// </summary>
        private new float[] lp_E;
        /// <summary>
        /// raw time delay (in milliseconds) of local grouped excitations
        /// </summary>
        private float raw_tau_G;
        /// <summary>
        /// dynamic time delay of (in milliseconds) of local grouped excitations
        /// </summary>
        private float dyn_tau_G;
        /// <summary>
        /// dynamic delay coefficient of local grouped excitations
        /// </summary>
        private float dyn_lp_G;
        /// <summary>
        /// pre-synaptically spatial bias connections matrix
        /// </summary>
        private float[,] BiasMat;
        /// <summary>
        /// standard deviation in normal-distributed spatial weighting matrix
        /// </summary>
        private float std_w;
        /// <summary>
        /// standard deviation in normal-distributed spatial weighting matrix
        /// </summary>
        private float std_h;
        /// <summary>
        /// standard  deviation in Gaussian blur function
        /// </summary>
        private float std_gb;
        /// <summary>
        /// a scale parameter in making a Gaussian distribution of input stimuli
        /// </summary>
        private float gauss_scale_w;
        /// <summary>
        /// a scale parameter in making a Gaussian distribution of input stimuli
        /// </summary>
        private float gauss_scale_h;
        /// <summary>
        /// threshold of spiking rate
        /// </summary>
        private int Tsr;
        /// <summary>
        /// a scale for pre-synaptically spatial bias connections matrix
        /// </summary>
        private float biasScale;
        /// <summary>
        /// maximum likelihood in spatial bias connection matrix
        /// </summary>
        private float maxLikelihood;

        #endregion

        #region GA FIELD

        /// <summary>
        /// dictionary of searching parameters in GA
        /// </summary>
        private Dictionary<string, float> paramsDict;
        /// <summary>
        /// fitness value
        /// </summary>
        private float fitness;
        /// <summary>
        /// old (true) or new (false) generation of object
        /// </summary>
        private bool isOldGeneration;
        /// <summary>
        /// timing of activation of neuron
        /// </summary>
        private int activationTiming;
        /// <summary>
        /// failure detection of colliding scenarios
        /// </summary>
        private int Fcol;
        /// <summary>
        /// wrong collision-like response to non-collision scenarios
        /// </summary>
        private int Fnon;
        /// <summary>
        /// belongs to best agents (true) or not (false)
        /// </summary>
        private bool isBestAgent;
        /// <summary>
        /// parent selected (true) or not (false)
        /// </summary>
        private bool isParent;
        /// <summary>
        /// max value of spiking rate threshold
        /// </summary>
        public readonly int max_Tsr;
        /// <summary>
        /// min value of spiking rate threshold
        /// </summary>
        public readonly int min_Tsr;
        /// <summary>
        /// max value of FFI threshold
        /// </summary>
        public new readonly int max_Tffi;
        /// <summary>
        /// min value of FFI threshold
        /// </summary>
        public new readonly int min_Tffi;
        /// <summary>
        /// max value of spiking threshold
        /// </summary>
        public new readonly float max_Tsp;
        /// <summary>
        /// min value of spiking threshold
        /// </summary>
        public new readonly float min_Tsp;
        /// <summary>
        /// min value of decay threshold
        /// </summary>
        public readonly int min_Tde;
        /// <summary>
        /// max value of decay threshold
        /// </summary>
        public readonly int max_Tde;
        /// <summary>
        /// max value of gaussian distributed spatial inhibitory connection matrix
        /// </summary>
        public readonly float max_std_w;
        /// <summary>
        /// min value of gaussian distributed spatial inhibitory connection matrix
        /// </summary>
        public readonly float min_std_w;
        /// <summary>
        /// min value of time constant in high-pass SFA mechanism
        /// </summary>
        public new readonly int min_tau_hp;
        /// <summary>
        /// max value of time constant in high-pass SFA mechanism
        /// </summary>
        public new readonly int max_tau_hp;
        /// <summary>
        /// max value of coefficient in sigmoid transformation
        /// </summary>
        public new readonly float max_Csig;
        /// <summary>
        /// min value of coefficient in sigmoid transformation
        /// </summary>
        public new readonly float min_Csig;
        /// <summary>
        /// min value of baseline of local inhibitory bias
        /// </summary>
        public readonly float min_W_base;
        /// <summary>
        /// max value of baseline of local inhibitory bias
        /// </summary>
        public readonly float max_W_base;
        /// <summary>
        /// min value of time constant in low-pass filtering of local excitations (central part of convolution kernel)
        /// </summary>
        public readonly int min_tau_cen_E;
        /// <summary>
        /// max value of time constant in low-pass filtering of local excitations (central part of convolution kernel)
        /// </summary>
        public readonly int max_tau_cen_E;

        #endregion

        #region NOISE FIELD

        /// <summary>
        /// random number generator
        /// </summary>
        private static Random _rand;
        /// <summary>
        /// standard deviation in gaussian density function
        /// </summary>
        private readonly float gauss_sigma;
        /// <summary>
        /// expectation in gaussian density function
        /// </summary>
        private readonly float gauss_mu;
        /// <summary>
        /// a probability in salt-and-pepper noise generation
        /// </summary>
        private readonly float P1;
        /// <summary>
        /// a probability in salt-and-pepper noise generation
        /// </summary>
        private readonly float P2;
        /// <summary>
        /// a probability in salt-and-pepper noise generation
        /// </summary>
        private readonly float P3;
        /// <summary>
        /// a scale in adding gaussian noise
        /// </summary>
        private readonly byte gauss_scale;

        #endregion

        #region LGMD PROPERTY

        /// <summary>
        /// property of gauss blurred motion
        /// </summary>
        public float[,] GBM
        {
            get { return blurred; }
            set { blurred = value; }
        }

        /// <summary>
        /// property of dynamic delay time in milliseconds in G layer
        /// </summary>
        public float DYN_TAU_G
        {
            get { return dyn_tau_G; }
            set { dyn_tau_G = value; }
        }

        /// <summary>
        /// property of spiking rate threshold evoking 
        /// </summary>
        public int TSR
        {
            get { return this.Tsr; }
            set { this.Tsr = value; }
        }

        /// <summary>
        /// property of FFI threshold
        /// </summary>
        public new int TFFI
        {
            get { return this.Tffi; }
            set { this.Tffi = value; }
        }

        /// <summary>
        /// property of decay threshold
        /// </summary>
        public int TDE
        {
            get { return this.Tde; }
            set { this.Tde = value; }
        }

        /// <summary>
        /// property of spiking threshold
        /// </summary>
        public new float TSP
        {
            get { return this.Tsp; }
            set { this.Tsp = value; }
        }

        /// <summary>
        /// property of coefficient in sigmoid transformation
        /// </summary>
        public new float COE_SIG
        {
            get { return this.coe_sig; }
            set { this.coe_sig = value; }
        }

        /// <summary>
        /// property of baseline of local inhibitory bias
        /// </summary>
        public float W_BASE
        {
            get { return this.W_base; }
            set { this.W_base = value; }
        }

        /// <summary>
        /// property of standard deviation in gaussian-distributed lateral inhibitory connection matrix
        /// </summary>
        public float STD_W
        {
            get { return this.std_w; }
            set { this.std_w = value; }
        }

        /// <summary>
        /// property of time constant in SFA highpass filter
        /// </summary>
        public new int TAU_HP
        {
            get { return this.tau_hp; }
            set { this.tau_hp = value; }
        }

        /// <summary>
        /// property of time constant in convolution of local excitations
        /// </summary>
        public float TAU_CEN_E
        {
            get { return this.tau_E[0]; }
            set { this.tau_E[0] = value; }
        }

        #endregion

        #region GA PROPERTY

        /// <summary>
        /// property of searching parameters dictionary
        /// </summary>
        public new Dictionary<string, float> ParamsDict
        {
            get { return paramsDict; }
            set { paramsDict = value; }
        }

        /// <summary>
        /// property of neuron activating timing
        /// </summary>
        public new int ActivationTiming
        {
            get { return activationTiming; }
            set { activationTiming = value; }
        }

        /// <summary>
        /// property of fitness value
        /// </summary>
        public new float Fitness
        {
            get { return fitness; }
            set { fitness = value; }
        }

        /// <summary>
        /// property of old generation checking
        /// </summary>
        public new bool IsOldGeneration
        {
            get { return isOldGeneration; }
            set { isOldGeneration = value; }
        }

        /// <summary>
        /// property of failure of collision detection
        /// </summary>
        public new int FCOL
        {
            get { return Fcol; }
            set { Fcol = value; }
        }

        /// <summary>
        /// property of wrong response to non-collision scenarios
        /// </summary>
        public new int FNON
        {
            get { return Fnon; }
            set { Fnon = value; }
        }

        /// <summary>
        /// property of best agent
        /// </summary>
        public new bool IsBestAgent
        {
            get { return isBestAgent; }
            set { isBestAgent = value; }
        }

        /// <summary>
        /// property of parent selection
        /// </summary>
        public new bool IsParent
        {
            get { return isParent; }
            set { isParent = value; }
        }

        #endregion

        #region CONSTRUCTOR
        
        /// <summary>
        /// Constructor
        /// </summary>
        public LGMDPlus()
        { }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="fps"></param>
        public LGMDPlus(int width, int height, int fps) : base(width, height, fps)
        {
            blurred = new float[height, width];
            gauss_blur_kernel_width = 3;
            std_gb = 1;
            gauss_blur_kernel = new float[gauss_blur_kernel_width, gauss_blur_kernel_width];
            //gene: std_w
            std_w = 0.5307723f;
            std_h = std_w * height / width;
            //gene: Tffi
            Tffi = 12;
            //gene: W_base
            W_base = 0.6190473f;
            //gene: tau_hp
            tau_hp = 783; //ms
            hp_delay = this.tau_hp / (time_interval + tau_hp);
            ConvK = new float[2 * Np + 1, 2 * Np + 1];
            tau_E = new float[2 * Np + 1];
            lp_E = new float[2 * Np + 1];
            W_min = 0.1f;
            gauss_scale_w = 3 * std_w;
            gauss_scale_h = 3 * std_h;
            BiasMat = new float[height, width];
            MakeGaussian(std_h, std_w, width, height, gauss_scale_w, gauss_scale_h, ref BiasMat, out biasScale, out maxLikelihood);
            MakeInhibitoryKernel(Np, ref ConvK);
            base.MakeGaussian(std_gb, gauss_blur_kernel_width, ref gauss_blur_kernel);
            //gene: tau_E[0]
            for (int i = 0; i <  2 * Np + 1; i++)
            {
                tau_E[i] = 10 + i * 10; //ms
                lp_E[i] = base.time_interval / (base.time_interval + tau_E[i]);
            }
            raw_tau_G = 10; //ms
            dyn_tau_G = raw_tau_G;
            dyn_lp_G = base.time_interval / (base.time_interval + dyn_tau_G);
            //gene: coe_sig
            coe_sig = 0.6f;
            //gene: Tsp
            Tsp = 0.68f;
            clip_point = 0.1f;
            Nts = 10;
            spike = new byte[Nts];
            dc = 0.1f;
            W_on = 2;
            W_off = 1;
            //gene: Tsr
            Tsr = 56;
            //gene: Tde
            Tde = 18;
            collision = 0;

            //attention for GA parameters initialisation
            isOldGeneration = false;
            fitness = 0;
            activationTiming = 0;
            isBestAgent = false;
            isParent = false;
            Fcol = 0;
            Fnon = 0;
            max_Tsr = 150;
            min_Tsr = 20;
            max_Tffi = 30;
            min_Tffi = 5;
            max_Tsp = 0.9f;
            min_Tsp = 0.6f;
            max_Tde = 50;
            min_Tde = 5;
            max_tau_hp = 1300;
            min_tau_hp = 300;
            max_tau_cen_E = 50;
            min_tau_cen_E = 1;
            max_W_base = 2.0f;
            min_W_base = 0.1f;
            max_Csig = 2.0f;
            min_Csig = 0.1f;
            max_std_w = 2.0f;
            min_std_w = 0.1f;
            //attention
            paramsDict = new Dictionary<string, float>();
            paramsDict.Add("Tsr", this.Tsr);
            paramsDict.Add("Tffi", this.Tffi);
            paramsDict.Add("Tsp", this.Tsp);
            paramsDict.Add("Tde", this.Tde);
            paramsDict.Add("std_w", this.std_w);
            paramsDict.Add("W_base", this.W_base);
            paramsDict.Add("coe_sig", this.coe_sig);
            paramsDict.Add("tau_hp", this.tau_hp);
            paramsDict.Add("tau_E", this.tau_E[0]);

            //attention for noise testing initialisation
            _rand = new Random(GetRandomSeed());
            gauss_sigma = 1;
            gauss_mu = 0;
            P1 = 0.2f;
            P2 = 0.2f;
            P3 = P2 / (1 - P1);
            gauss_scale = 64;

            Console.WriteLine("LGMDPlus visual neural network parameters initialisation completed.....\n");
        }

        #endregion

        #region ALGORITHM

        /// <summary>
        /// Make gauss filter kernel
        /// </summary>
        /// <param name="sigma"></param>
        /// <param name="gauss_width"></param>
        /// <param name="scale"></param>
        /// <param name="gauss_kernel"></param>
        private void MakeGaussian(float sigma, int gauss_width, float scale, ref float[,] gauss_kernel)
        {
            //centriod
            int centroid = gauss_width / 2;
            float gaussian;
            float distance;
            float scale_x;
            float scale_y;

            for (int i = 0; i < gauss_width; i++)
            {
                for (int j = 0; j < gauss_width; j++)
                {
                    //attention
                    //scale x and y within the defined mean-sigma related range of gaussian distribution density function
                    scale_x = (i - centroid) * scale / centroid;
                    scale_y = (j - centroid) * scale / centroid;
                    distance = scale_x * scale_x + scale_y * scale_y;
                    gaussian = (float)(Math.Exp((0 - distance) / (2 * sigma * sigma)) / (2 * Math.PI * sigma * sigma));
                    gauss_kernel[i, j] = gaussian;
                }
            }
        }

        /// <summary>
        /// Make gauss filter kernel
        /// </summary>
        /// <param name="sigma1"></param>
        /// <param name="sigma2"></param>
        /// <param name="gauss_width"></param>
        /// <param name="gauss_height"></param>
        /// <param name="scale_w"></param>
        /// <param name="scale_h"></param>
        /// <param name="gauss_kernel"></param>
        /// <param name="biasScale"></param>
        /// <param name="maxL"></param>
        private void MakeGaussian(float sigma1, float sigma2, int gauss_width, int gauss_height, float scale_w, float scale_h, ref float[,] gauss_kernel, out float biasScale, out float maxL)
        {
            biasScale = 0;
            maxL = 0;
            //centriod
            int centroidX = gauss_height / 2;
            int centroidY = gauss_width / 2;
            float gaussian;
            //float distance;
            float distance1;
            float distance2;
            float scale_x;
            float scale_y;

            for (int i = 0; i < gauss_height; i++)
            {
                for (int j = 0; j < gauss_width; j++)
                {
                    //attention
                    //scale x and y within the defined mean-sigma related range of gaussian distribution density function
                    scale_x = (i - centroidX) * scale_h / centroidX;
                    scale_y = (j - centroidY) * scale_w / centroidY;
                    //distance = scale_x * scale_x + scale_y * scale_y;
                    distance1 = scale_x * scale_x;
                    distance2 = scale_y * scale_y;
                    gaussian = (float)(Math.Exp(-0.5 * ((distance1 / (sigma1 * sigma1)) + (distance2 / (sigma2 * sigma2)))) / (2 * Math.PI * sigma1 * sigma2));
                    //gaussian = (float)(Math.Exp((0 - distance) / (2 * sigma1 * sigma2)) / (2 * Math.PI * sigma1 * sigma2));
                    gauss_kernel[i, j] = gaussian;
                    if (maxL < gaussian)
                        maxL = gaussian;
                    biasScale += gaussian;
                }
            }
            /*
            for (int i = 0; i < gauss_height; i++)
            {
                for (int j = 0; j < gauss_width; j++)
                {
                    gauss_kernel[i, j] /= biasScale;
                }
            }
            */
        }

        /// <summary>
        /// Make a inhibitory connection kernel
        /// </summary>
        /// <param name="Np"></param>
        /// <param name="mat"></param>
        private void MakeInhibitoryKernel(int Np, ref float[,] mat)
        {
            for (int i = -1; i < Np + 1; i++)
            {
                for (int j = -1; j < Np + 1; j++)
                {
                    if (i == 0 && j == 0)   //centre
                        mat[i + 1, j + 1] = 1;
                    else if (i == 0 || j == 0)  //nearest
                        mat[i + 1, j + 1] = 0.25f;
                    else   //diagonal
                        mat[i + 1, j + 1] = 0.125f;
                }
            }
        }

        /// <summary>
        /// Gauss blur
        /// </summary>
        /// <param name="inputMat"></param>
        /// <param name="outputMat"></param>
        /// <param name="gauss_width"></param>
        /// <param name="gauss_kernel"></param>
        /// <param name="t"></param>
        private float[,] GaussBlur(int[,,] inputMat, int gauss_width, float[,] gauss_kernel, int t)
        {
            float[,] outputMat = new float[height, width];
            int tmp = 0;
            //kernel radius
            int k_radius = gauss_width / 2;
            int r, c;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    for (int i = -k_radius; i < k_radius + 1; i++)
                    {
                        for (int j = -k_radius; j < k_radius + 1; j++)
                        {
                            r = y + i;
                            c = x + j;
                            //if exceeding range, let it equal to near pixel
                            while (r < 0)
                            { r++; }
                            while (r >= height)
                            { r--; }
                            while (c < 0)
                            { c++; }
                            while (c >= width)
                            { c--; }
                            //************************
                            tmp = (int)(inputMat[r, c, t] * gauss_kernel[i + k_radius, j + k_radius]);
                            outputMat[y, x] += tmp;
                        }
                    }
                }
            }
            return outputMat;
        }

        /// <summary>
        /// Collision detection
        /// </summary>
        /// <param name="sr"></param>
        /// <param name="tsr"></param>
        /// <returns></returns>
        private byte collisionDetecting(float sr, int tsr)
        {
            if (sr >= tsr)
                return 1;
            else
                return 0;
        }

        #endregion

        #region NOISE TESTING FUNCTION

        /// <summary>
        /// Control speed of getting random seed
        /// </summary>
        /// <returns></returns>
        private static int GetRandomSeed()
        {
            byte[] bytes = new byte[4];
            System.Security.Cryptography.RNGCryptoServiceProvider rng = new System.Security.Cryptography.RNGCryptoServiceProvider();
            rng.GetBytes(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        /// <summary>
        /// Box-Muller transform to generation a standard normal distributed random number as gaussian noise
        /// </summary>
        /// <returns></returns>
        private double GaussNoise(float sigma, float mu, byte scale)
        {
            double r1 = _rand.NextDouble();
            double r2 = _rand.NextDouble();
            double result = Math.Sqrt((-2) * Math.Log(r2)) * Math.Sin(2 * Math.PI * r1);
            return (result * sigma + mu) * scale;
        }

        /// <summary>
        /// Gaussian density function
        /// </summary>
        /// <param name="rand"></param>
        /// <param name="mu"></param>
        /// <param name="sigma"></param>
        /// <returns></returns>
        private double GaussDensity(float x, float mu, float sigma)
        {
            double p;
            p = (1 / (sigma * Math.Sqrt(2 * Math.PI)) * Math.Pow(Math.E, -(Math.Pow(x - mu, 2) / (2 * sigma * sigma))));
            return p;
        }

        /// <summary>
        /// Method to add salt-and-pepper noise
        /// </summary>
        /// <param name="Pa"></param>
        /// <param name="Pb"></param>
        /// <returns></returns>
        private byte AddSaltAndPepper(double Pa, double Pb, byte gray)
        {
            double likelihood1 = _rand.NextDouble();
            if (likelihood1 < Pa)
                return 255;
            else
            {
                double likelihood2 = _rand.NextDouble();
                if (likelihood2 < Pb)
                    return 0;
            }
            return gray;
        }

        #endregion

        #region LGMDPlus MODEL VISUAL PROCESSING
        /// <summary>
        /// LGMDPlus model visual processing
        /// </summary>
        /// <param name="img1"></param>
        /// <param name="img2"></param>
        /// <param name="t"></param>
        public void LGMDPlus_Processing(byte[,,] img1, byte[,,] img2, int t)
        {
            float tmp_sum = 0;
            float tmp_ffi = 0;
            int cur_frame = t % ONs.GetLength(2);
            int pre_frame = (t - 1) % ONs.GetLength(2);
            int cur_spi = t % spike.Length;
            float sigma, sca, localW;

            //PHOTORECEPTORS
            for (int y = 0; y < this.height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    photoreceptors[y, x, cur_frame] = HighpassFilter(img1[y, x, 0], img2[y, x, 0]);
                    tmp_ffi += Math.Abs(photoreceptors[y, x, cur_frame]);
                }
            }

            //attention
            //SPATIAL LOW-PASS FILTERING
            blurred = GaussBlur(photoreceptors, gauss_blur_kernel_width, gauss_blur_kernel, cur_frame);

            //attention
            //FFI TUNING
            ffi[cur_frame] = tmp_ffi / Ncell;
            ffi[cur_frame] = LowpassFilter(ffi[cur_frame], ffi[pre_frame], lp_FFI);
            sigma = FFIshapesDelayCoefficient(ffi[cur_frame]);
            temporalTuning(raw_tau_G, ref dyn_tau_G, ref dyn_lp_G, sigma);
            dyn_bias = FFIMediation(ffi[cur_frame], W_base);

            //ON/OFF MECHANISMS
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    ons[y, x, cur_frame] = HRplusDC_ON(ons[y, x, pre_frame], blurred[y, x]);
                    offs[y, x, cur_frame] = HRplusDC_OFF(offs[y, x, pre_frame], blurred[y, x]);
                }
            }

            //SPATIOTEMPORAL PROCESSING IN DUAL-PATHWAYS
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    localW = 1 - BiasMat[y, x];
                    if (localW < W_min)
                        localW = W_min;
                    Inh_ON[y, x] = Convolution(y, x, ons, ConvK, cur_frame, pre_frame, lp_E);
                    Inh_OFF[y, x] = Convolution(y, x, offs, ConvK, cur_frame, pre_frame, lp_E);
                    //attention
                    //dyn_bias = FFIMediation(ffi[cur_frame], (1 - BiasMat[y, x]));
                    //S_on = sCellValue(ons[y, x, cur_frame], Inh_ON[y, x], dyn_bias);
                    //S_off = sCellValue(offs[y, x, cur_frame], Inh_OFF[y, x], dyn_bias);
                    S_on = sCellValue(ons[y, x, cur_frame], Inh_ON[y, x], localW * dyn_bias);
                    S_off = sCellValue(offs[y, x, cur_frame], Inh_OFF[y, x], localW * dyn_bias);
                    //S_on = sCellValue(ons[y, x, cur_frame] * BiasMat[y, x], Inh_ON[y, x], dyn_bias);
                    //S_off = sCellValue(offs[y, x, cur_frame] * BiasMat[y, x], Inh_OFF[y, x], dyn_bias);
                    scells[y, x] = SupralinearSummation(S_on, S_off);
                }
            }
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    gcells[y, x, cur_frame] = Convolving(y, x, scells, W_g);
                }
            }
            sca = Scale(cur_frame);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    gcells[y, x, cur_frame] = gCellValue(scells[y, x], gcells[y, x, cur_frame], sca);
                    //latency of input local excitation
                    gcells[y, x, cur_frame] = LowpassFilter(gcells[y, x, cur_frame], gcells[y, x, pre_frame], dyn_lp_G);
                    tmp_sum += gcells[y, x, cur_frame];
                }
            }

            //MEMBRANE POTENTIAL
            mp[cur_frame] = tmp_sum;
            /*
            if (ffi[cur_frame] > 1)
                mp[cur_frame] /= (float)(Math.Sqrt(ffi[cur_frame]));
            */
            smp[cur_frame] = SigmoidTransfer(mp[cur_frame]);

            //SPIKE FREQUENCY ADAPTATION
            sfa[cur_frame] = SFA_HPF(sfa[pre_frame], smp[pre_frame], smp[cur_frame]);

            //SPIKING
            spike[cur_spi] = Spiking(sfa[cur_frame]);

            //RATE
            spiRate = spikeFrequency(spike);

            //Collision Recognition
            if (collision == 0)
            {
                collision = collisionDetecting(spiRate, Tsr);
                //attention
                if (collision == 1)
                    activationTiming = t;
            }

            //Console.WriteLine("{0} {1:F} {2:F} {3:F} {4:F} {5} {6:F} {7:F} {8:F}", t, mp[cur_frame], smp[cur_frame], sfa[cur_frame], ffi[cur_frame], spike[cur_spi], spiRate, dyn_bias, dyn_tau_G);
            Console.WriteLine("{0} {1:F} {2:F} {3:F} {4:F} {5} {6:F} {7}", t, mp[cur_frame], smp[cur_frame], sfa[cur_frame], ffi[cur_frame], spike[cur_spi], spiRate, collision);

        }

        /// <summary>
        /// Exchange searching parameters to object domain (LGMDPlus model)
        /// </summary>
        public void LGMDPlus_searchingParametersExchanging()
        {
            Tffi = (int)paramsDict["Tffi"];
            Tde = (int)paramsDict["Tde"];
            Tsp = paramsDict["Tsp"];
            Tsr = (int)paramsDict["Tsr"];
            tau_E[0] = (int)paramsDict["tau_E"];
            lp_E[0] = base.time_interval / (base.time_interval + tau_E[0]);
            for (int i = 1; i < 2 * Np + 1; i++)
            {
                tau_E[i] = tau_E[0] + i * tau_E[0]; //ms
                lp_E[i] = base.time_interval / (base.time_interval + tau_E[i]);
            }
            tau_hp = (int)paramsDict["tau_hp"];
            hp_delay = tau_hp / (tau_hp + time_interval);
            W_base = paramsDict["W_base"];
            coe_sig = paramsDict["coe_sig"];
            std_w = paramsDict["std_w"];
            std_h = std_w * height / width;
            gauss_scale_w = 3 * std_w;
            gauss_scale_h = 3 * std_h;
            BiasMat = new float[height, width];
            MakeGaussian(std_h, std_w, width, height, gauss_scale_w, gauss_scale_h, ref BiasMat, out biasScale, out maxLikelihood);
        }

        /// <summary>
        /// LGMDPlus mode resetting
        /// </summary>
        public void LGMDPlus_Reset()
        {
            for (int i = 0; i < ffi.Length; i++)
            {
                ffi[i] = 0;
                mp[i] = 0;
                smp[i] = 0;
            }
            spiRate = 0;
            for (int i = 0; i < Nts; i++)
            {
                spike[i] = 0;
            }
            photoreceptors = new int[height, width, 2];
            ons = new float[height, width, 2];
            offs = new float[height, width, 2];
            gcells = new float[height, width, 2];
        }

        /// <summary>
        /// Overwrite CompareTo method in IComparable
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public new int CompareTo(Object obj)
        {
            if (this.fitness > ((LGMDPlus)obj).fitness)
                return -1;
            else if (this.fitness == ((LGMDPlus)obj).fitness)
                return 0;
            else
                return 1;
        }

        /// <summary>
        /// Overwrite Compare method in IComparer<LGMDPlus>
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public int Compare(LGMDPlus a, LGMDPlus b)
        {
            if (a.fitness < b.fitness)
                return -1;
            else if (a.fitness == b.fitness)
                return 0;
            else
                return 1;
        }

        /// <summary>
        /// LGMDPlus model visual processing
        /// </summary>
        /// <param name="img1"></param>
        /// <param name="img2"></param>
        /// <param name="t"></param>
        public void LGMDPlus_Gauss_Noise_Testing(byte[,,] img1, byte[,,] img2, int t)
        {
            float tmp_sum = 0;
            float tmp_ffi = 0;
            int cur_frame = t % ONs.GetLength(2);
            int pre_frame = (t - 1) % ONs.GetLength(2);
            int cur_spi = t % spike.Length;
            float sigma, sca, localW;

            //PHOTORECEPTORS
            for (int y = 0; y < this.height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    //attention: add pixelwise gaussian noise to input 
                    double gn1 = GaussNoise(this.gauss_sigma, this.gauss_mu, this.gauss_scale);
                    int tmp1 = img1[y, x, 0] + (int)gn1;
                    if (tmp1 > 255)
                        img1[y, x, 0] = 255;
                    else if (tmp1 < 0)
                        img1[y, x, 0] = 0;
                    else
                        img1[y, x, 0] = (byte)tmp1;
                    double gn2 = GaussNoise(this.gauss_sigma, this.gauss_mu, this.gauss_scale);
                    int tmp2 = img2[y, x, 0] + (int)gn2;
                    if (tmp2 > 255)
                        img2[y, x, 0] = 255;
                    else if (tmp2 < 0)
                        img2[y, x, 0] = 0;
                    else
                        img2[y, x, 0] = (byte)tmp2;

                    photoreceptors[y, x, cur_frame] = HighpassFilter(img1[y, x, 0], img2[y, x, 0]);
                    tmp_ffi += Math.Abs(photoreceptors[y, x, cur_frame]);
                }
            }

            //attention
            //SPATIAL LOW-PASS FILTERING
            blurred = GaussBlur(photoreceptors, gauss_blur_kernel_width, gauss_blur_kernel, cur_frame);

            //attention
            //FFI TUNING
            ffi[cur_frame] = tmp_ffi / Ncell;
            ffi[cur_frame] = LowpassFilter(ffi[cur_frame], ffi[pre_frame], lp_FFI);
            sigma = FFIshapesDelayCoefficient(ffi[cur_frame]);
            temporalTuning(raw_tau_G, ref dyn_tau_G, ref dyn_lp_G, sigma);
            dyn_bias = FFIMediation(ffi[cur_frame], W_base);

            //ON/OFF MECHANISMS
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    ons[y, x, cur_frame] = HRplusDC_ON(ons[y, x, pre_frame], blurred[y, x]);
                    offs[y, x, cur_frame] = HRplusDC_OFF(offs[y, x, pre_frame], blurred[y, x]);
                }
            }

            //SPATIOTEMPORAL PROCESSING IN DUAL-PATHWAYS
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    localW = 1 - BiasMat[y, x];
                    if (localW < W_min)
                        localW = W_min;
                    Inh_ON[y, x] = Convolution(y, x, ons, ConvK, cur_frame, pre_frame, lp_E);
                    Inh_OFF[y, x] = Convolution(y, x, offs, ConvK, cur_frame, pre_frame, lp_E);
                    //attention
                    //dyn_bias = FFIMediation(ffi[cur_frame], (1 - BiasMat[y, x]));
                    //S_on = sCellValue(ons[y, x, cur_frame], Inh_ON[y, x], dyn_bias);
                    //S_off = sCellValue(offs[y, x, cur_frame], Inh_OFF[y, x], dyn_bias);
                    S_on = sCellValue(ons[y, x, cur_frame], Inh_ON[y, x], localW * dyn_bias);
                    S_off = sCellValue(offs[y, x, cur_frame], Inh_OFF[y, x], localW * dyn_bias);
                    //S_on = sCellValue(ons[y, x, cur_frame] * BiasMat[y, x], Inh_ON[y, x], dyn_bias);
                    //S_off = sCellValue(offs[y, x, cur_frame] * BiasMat[y, x], Inh_OFF[y, x], dyn_bias);
                    scells[y, x] = SupralinearSummation(S_on, S_off);
                }
            }
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    gcells[y, x, cur_frame] = Convolving(y, x, scells, W_g);
                }
            }
            sca = Scale(cur_frame);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    gcells[y, x, cur_frame] = gCellValue(scells[y, x], gcells[y, x, cur_frame], sca);
                    //latency of input local excitation
                    gcells[y, x, cur_frame] = LowpassFilter(gcells[y, x, cur_frame], gcells[y, x, pre_frame], dyn_lp_G);
                    tmp_sum += gcells[y, x, cur_frame];
                }
            }

            //MEMBRANE POTENTIAL
            mp[cur_frame] = tmp_sum;
            /*
            if (ffi[cur_frame] > 1)
                mp[cur_frame] /= (float)(Math.Sqrt(ffi[cur_frame]));
            */
            smp[cur_frame] = SigmoidTransfer(mp[cur_frame]);

            //SPIKE FREQUENCY ADAPTATION
            sfa[cur_frame] = SFA_HPF(sfa[pre_frame], smp[pre_frame], smp[cur_frame]);

            //SPIKING
            spike[cur_spi] = Spiking(sfa[cur_frame]);

            //RATE
            spiRate = spikeFrequency(spike);

            //Collision Recognition
            if (collision == 0)
            {
                collision = collisionDetecting(spiRate, Tsr);
                //attention
                if (collision == 1)
                    activationTiming = t;
            }

            Console.WriteLine("{0} {1:F} {2:F} {3:F} {4:F} {5} {6:F} {7}", t, mp[cur_frame], smp[cur_frame], sfa[cur_frame], ffi[cur_frame], spike[cur_spi], spiRate, collision);

        }

        /// <summary>
        /// LGMDPlus model visual processing
        /// </summary>
        /// <param name="img1"></param>
        /// <param name="img2"></param>
        /// <param name="t"></param>
        public void LGMDPlus_SaltAndPepper_Noise_Testing(byte[,,] img1, byte[,,] img2, int t)
        {
            float tmp_sum = 0;
            float tmp_ffi = 0;
            int cur_frame = t % ONs.GetLength(2);
            int pre_frame = (t - 1) % ONs.GetLength(2);
            int cur_spi = t % spike.Length;
            float sigma, sca, localW;

            //PHOTORECEPTORS
            for (int y = 0; y < this.height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    //attention: add pixelwise gaussian noise to input 
                    double gn1 = GaussNoise(this.gauss_sigma, this.gauss_mu, this.gauss_scale);
                    int tmp1 = img1[y, x, 0] + (int)gn1;
                    if (tmp1 > 255)
                        img1[y, x, 0] = 255;
                    else if (tmp1 < 0)
                        img1[y, x, 0] = 0;
                    else
                        img1[y, x, 0] = (byte)tmp1;
                    double gn2 = GaussNoise(this.gauss_sigma, this.gauss_mu, this.gauss_scale);
                    int tmp2 = img2[y, x, 0] + (int)gn2;
                    if (tmp2 > 255)
                        img2[y, x, 0] = 255;
                    else if (tmp2 < 0)
                        img2[y, x, 0] = 0;
                    else
                        img2[y, x, 0] = (byte)tmp2;

                    photoreceptors[y, x, cur_frame] = HighpassFilter(img1[y, x, 0], img2[y, x, 0]);
                    tmp_ffi += Math.Abs(photoreceptors[y, x, cur_frame]);
                }
            }

            //attention
            //SPATIAL LOW-PASS FILTERING
            blurred = GaussBlur(photoreceptors, gauss_blur_kernel_width, gauss_blur_kernel, cur_frame);

            //attention
            //FFI TUNING
            ffi[cur_frame] = tmp_ffi / Ncell;
            ffi[cur_frame] = LowpassFilter(ffi[cur_frame], ffi[pre_frame], lp_FFI);
            sigma = FFIshapesDelayCoefficient(ffi[cur_frame]);
            temporalTuning(raw_tau_G, ref dyn_tau_G, ref dyn_lp_G, sigma);
            dyn_bias = FFIMediation(ffi[cur_frame], W_base);

            //ON/OFF MECHANISMS
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    ons[y, x, cur_frame] = HRplusDC_ON(ons[y, x, pre_frame], blurred[y, x]);
                    offs[y, x, cur_frame] = HRplusDC_OFF(offs[y, x, pre_frame], blurred[y, x]);
                }
            }

            //SPATIOTEMPORAL PROCESSING IN DUAL-PATHWAYS
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    localW = 1 - BiasMat[y, x];
                    if (localW < W_min)
                        localW = W_min;
                    Inh_ON[y, x] = Convolution(y, x, ons, ConvK, cur_frame, pre_frame, lp_E);
                    Inh_OFF[y, x] = Convolution(y, x, offs, ConvK, cur_frame, pre_frame, lp_E);
                    //attention
                    //dyn_bias = FFIMediation(ffi[cur_frame], (1 - BiasMat[y, x]));
                    //S_on = sCellValue(ons[y, x, cur_frame], Inh_ON[y, x], dyn_bias);
                    //S_off = sCellValue(offs[y, x, cur_frame], Inh_OFF[y, x], dyn_bias);
                    S_on = sCellValue(ons[y, x, cur_frame], Inh_ON[y, x], localW * dyn_bias);
                    S_off = sCellValue(offs[y, x, cur_frame], Inh_OFF[y, x], localW * dyn_bias);
                    //S_on = sCellValue(ons[y, x, cur_frame] * BiasMat[y, x], Inh_ON[y, x], dyn_bias);
                    //S_off = sCellValue(offs[y, x, cur_frame] * BiasMat[y, x], Inh_OFF[y, x], dyn_bias);
                    scells[y, x] = SupralinearSummation(S_on, S_off);
                }
            }
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    gcells[y, x, cur_frame] = Convolving(y, x, scells, W_g);
                }
            }
            sca = Scale(cur_frame);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    gcells[y, x, cur_frame] = gCellValue(scells[y, x], gcells[y, x, cur_frame], sca);
                    //latency of input local excitation
                    gcells[y, x, cur_frame] = LowpassFilter(gcells[y, x, cur_frame], gcells[y, x, pre_frame], dyn_lp_G);
                    tmp_sum += gcells[y, x, cur_frame];
                }
            }

            //MEMBRANE POTENTIAL
            mp[cur_frame] = tmp_sum;
            /*
            if (ffi[cur_frame] > 1)
                mp[cur_frame] /= (float)(Math.Sqrt(ffi[cur_frame]));
            */
            smp[cur_frame] = SigmoidTransfer(mp[cur_frame]);

            //SPIKE FREQUENCY ADAPTATION
            sfa[cur_frame] = SFA_HPF(sfa[pre_frame], smp[pre_frame], smp[cur_frame]);

            //SPIKING
            spike[cur_spi] = Spiking(sfa[cur_frame]);

            //RATE
            spiRate = spikeFrequency(spike);

            //Collision Recognition
            if (collision == 0)
            {
                collision = collisionDetecting(spiRate, Tsr);
                //attention
                if (collision == 1)
                    activationTiming = t;
            }

            Console.WriteLine("{0} {1:F} {2:F} {3:F} {4:F} {5} {6:F} {7}", t, mp[cur_frame], smp[cur_frame], sfa[cur_frame], ffi[cur_frame], spike[cur_spi], spiRate, collision);

        }

        #endregion
    }
}

