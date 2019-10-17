/*
 * Filename: LGMDSingle.cs
 * Author: Qinbing FU
 * Location: Guangzhou
 * Date: May 2019
 */


using System;
using System.Collections.Generic;


namespace LGMD
{
    /// <summary>
    /// LGMD1 visual neural network with a single processing pathway (reference to the TNN paper authored by Shigang Yue)
    /// This model has two cells of LGMD and FFI, as well as four layers of photoreceptor (P) layer, excitation (E) layer, inhibition (I) layer, summation (S) layer.
    /// 
    /// References: [1] S. Yue, F. C. Rind, "Collision Detection in Complex Dynamic Scenes Using a LGMD Based Visual Neural Network with Feature Enhancement", IEEE Transactions on Neural Networks, 2006.
    ///             [2] S. Yue, F. C. Rind, "Redundant Neural Vision Systems - Competing for collision Recognition Roles", IEEE Transactions on Autonomous Mental Development, 2013.
    /// </summary>
    internal class LGMDSingle : IComparable, IComparer<LGMDSingle>
    {
        #region LGMD FIELD

        /// <summary>
        /// local inhibition bias
        /// </summary>
        protected float Wi;
        /// <summary>
        /// local threshold in S layer
        /// </summary>
        protected int Ts;
        /// <summary>
        /// spiking threshold
        /// </summary>
        protected float Tsp;
        /// <summary>
        /// inhibiting area radius
        /// </summary>
        protected byte Np;
        /// <summary>
        /// FFI threshold
        /// </summary>
        protected int Tffi;
        /// <summary>
        /// total number of cells in each pre-synaptic layer
        /// </summary>
        protected int Ncell;
        /// <summary>
        /// number of successive spikes
        /// </summary>
        protected byte Nsp;
        /// <summary>
        /// number of successive discrete time steps
        /// </summary>
        protected byte Nts;
        /// <summary>
        /// coefficient in sigmoid transformation
        /// </summary>
        protected float Csig;
        /// <summary>
        /// kernel in convolution of excitations forming inhibitions
        /// </summary>
        protected float[,] WI;
        /// <summary>
        /// photoreceptor layer matrix
        /// </summary>
        protected int[,] Photoreceptors;
        /// <summary>
        /// excitation layer matrix
        /// </summary>
        protected int[,,] Excitations;
        /// <summary>
        /// inhibition layer matrix
        /// </summary>
        protected float[,] Inhibitions;
        /// <summary>
        /// summation layer matrix
        /// </summary>
        protected float[,] Summations;
        /// <summary>
        /// membrane potential
        /// </summary>
        protected float mp;
        /// <summary>
        /// sigmoid membrane potential
        /// </summary>
        protected float smp;
        /// <summary>
        /// FFI value
        /// </summary>
        protected float ffi;
        /// <summary>
        /// spike value - spiking (1) or not (0)
        /// </summary>
        protected byte spike;
        /// <summary>
        /// collision event recognition (1) or not (0)
        /// </summary>
        protected byte collision;
        /// <summary>
        /// input frame width
        /// </summary>
        protected int width;
        /// <summary>
        /// input frame height
        /// </summary>
        protected int height;

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
        /// belongs to best agents (true) or not (false)
        /// </summary>
        private bool isBestAgent;
        /// <summary>
        /// parent selected (true) or not (false)
        /// </summary>
        private bool isParent;
        /// <summary>
        /// failure detection of colliding scenarios
        /// </summary>
        private int Fcol;
        /// <summary>
        /// wrong collision-like response to non-collision scenarios
        /// </summary>
        private int Fnon;
        /// <summary>
        /// max value of FFI threshold
        /// </summary>
        public readonly int max_Tffi;
        /// <summary>
        /// min value of FFI threshold
        /// </summary>
        public readonly int min_Tffi;
        /// <summary>
        /// max value of local threshold in S layer
        /// </summary>
        public readonly int max_Ts;
        /// <summary>
        /// min value of local threshold in S layer
        /// </summary>
        public readonly int min_Ts;
        /// <summary>
        /// max value of spiking threshold
        /// </summary>
        public readonly float max_Tsp;
        /// <summary>
        /// min value of spiking threshold
        /// </summary>
        public readonly float min_Tsp;
        /// <summary>
        /// max value of sigmoid transformation coefficient
        /// </summary>
        public readonly float max_Csig;
        /// <summary>
        /// min value of sigmoid transformation coefficient
        /// </summary>
        public readonly float min_Csig;
        /// <summary>
        /// max value of local inhibitory bias
        /// </summary>
        public readonly float max_Wi;
        /// <summary>
        /// min value of local inhibitory bias
        /// </summary>
        public readonly float min_Wi;

        #endregion

        #region LGMD PROPERTY

        /// <summary>
        /// property of neural membrane potential
        /// </summary>
        public float MP
        {
            get { return mp; }
            set { mp = value; }
        }

        /// <summary>
        /// property of sigmoid membrane potential
        /// </summary>
        public float SMP
        {
            get { return smp; }
            set { smp = value; }
        }

        /// <summary>
        /// property of spike
        /// </summary>
        public byte SPIKE
        {
            get { return spike; }
            set { spike = value; }
        }

        /// <summary>
        /// property of collision detection
        /// </summary>
        public byte COLLISION
        {
            get { return collision; }
            set { collision = value; }
        }

        /// <summary>
        /// property of photoreceptor layer
        /// </summary>
        public int[,] PHOTOS
        {
            get { return Photoreceptors; }
            set { Photoreceptors = value; }
        }

        /// <summary>
        /// property of FFI value
        /// </summary>
        public float FFI
        {
            get { return ffi; }
            set { ffi = value; }
        }

        /// <summary>
        /// property of excitation layer
        /// </summary>
        public int[,,] EXC
        {
            get { return Excitations; }
            set { Excitations = value; }
        }

        /// <summary>
        /// property of inhibition layer
        /// </summary>
        public float[,] INH
        {
            get { return Inhibitions; }
            set { Inhibitions = value; }
        }

        /// <summary>
        /// property of summation layer
        /// </summary>
        public float[,] SUM
        {
            get { return Summations; }
            set { Summations = value; }
        }

        /// <summary>
        /// property of FFI threshold
        /// </summary>
        public int TFFI
        {
            get { return Tffi; }
            set { Tffi = value; }
        }

        /// <summary>
        /// property of local threshold in S layer
        /// </summary>
        public int TS
        {
            get { return Ts; }
            set { Ts = value; }
        }

        /// <summary>
        /// property of spiking threshold
        /// </summary>
        public float TSP
        {
            get { return Tsp; }
            set { Tsp = value; }
        }

        /// <summary>
        /// property of coefficient in sigmoid transformation
        /// </summary>
        public float COE_SIG
        {
            get { return Csig; }
            set { Csig = value; }
        }

        /// <summary>
        /// property of local inhibition bias
        /// </summary>
        public float W_I
        {
            get { return Wi; }
            set { Wi = value; }
        }

        #endregion

        #region GA PROPERTY

        /// <summary>
        /// property of searching parameters dictionary
        /// </summary>
        public Dictionary<string, float> ParamsDict
        {
            get { return paramsDict; }
            set { paramsDict = value; }
        }

        /// <summary>
        /// property of neuron activating timing
        /// </summary>
        public int ActivationTiming
        {
            get { return activationTiming; }
            set { activationTiming = value; }
        }

        /// <summary>
        /// property of fitness value
        /// </summary>
        public float Fitness
        {
            get { return fitness; }
            set { fitness = value; }
        }

        /// <summary>
        /// property of old generation checking
        /// </summary>
        public bool IsOldGeneration
        {
            get { return isOldGeneration; }
            set { isOldGeneration = value; }
        }

        /// <summary>
        /// property of failure of collision detection
        /// </summary>
        public int FCOL
        {
            get { return Fcol; }
            set { Fcol = value; }
        }

        /// <summary>
        /// property of wrong response to non-collision scenarios
        /// </summary>
        public int FNON
        {
            get { return Fnon; }
            set { Fnon = value; }
        }

        /// <summary>
        /// property of best agent
        /// </summary>
        public bool IsBestAgent
        {
            get { return isBestAgent; }
            set { isBestAgent = value; }
        }

        /// <summary>
        /// property of parent selection
        /// </summary>
        public bool IsParent
        {
            get { return isParent; }
            set { isParent = value; }
        }

        #endregion

        #region CONSTRUCTOR

        /// <summary>
        /// Default constructor
        /// </summary>
        public LGMDSingle() { }

        /// <summary>
        /// Parameterized constructor
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        public LGMDSingle(int width /*frame width*/, int height /*frame height*/)
        {
            this.width = width;
            this.height = height;
            Ncell = width * height;
            Photoreceptors = new int[height, width];
            Excitations = new int[height, width, 2];
            Inhibitions = new float[height, width];
            Summations = new float[height, width];
            Np = 1;
            WI = new float[2 * Np + 1, 2 * Np + 1];
            //gene: Wi
            Wi = 0.6f;
            //gene: Ts
            Ts = 30;
            //gene: Tffi
            Tffi = 10;
            //gene: Tsp
            Tsp = 0.78f;
            Nsp = 0;
            Nts = 5;
            //gene: Csig
            Csig = 1;
            ffi = 0;
            mp = 0;
            smp = 0.5f;
            spike = 0;
            collision = 0;
            localIW(WI, Np);

            //attention for GA
            isOldGeneration = false;
            fitness = 0;
            activationTiming = 0;
            isBestAgent = false;
            isParent = false;
            Fcol = 0;
            Fnon = 0;
            max_Tffi = 30;
            min_Tffi = 5;
            max_Tsp = 0.91f;
            min_Tsp = 0.65f;
            max_Ts = 50;
            min_Ts = 10;
            max_Csig = 2.0f;
            min_Csig = 0.1f;
            max_Wi = 2.0f;
            min_Wi = 0.1f;
            //attention
            paramsDict = new Dictionary<string, float>();
            paramsDict.Add("Tffi", Tffi);
            paramsDict.Add("Ts", Ts);
            paramsDict.Add("Tsp", Tsp);
            paramsDict.Add("Wi", Wi);
            paramsDict.Add("coe_sig", Csig);

            Console.WriteLine("LGMD model parameters setting with a single visual processing pathway\n");
        }

        #endregion

        #region METHOD

        /// <summary>
        /// Constructing local convolution kernel
        /// </summary>
        /// <param name="mat"></param>
        private void localIW(float[,] mat, byte np)
        {
            for (int i = -1; i < np + 1; i++)
            {
                for (int j = -1; j < np + 1; j++)
                {
                    if (i == 0 && j == 0)
                        continue;
                    else if (i == 0 || j == 0)
                        mat[i + 1, j + 1] = 0.25f;
                    else
                        mat[i + 1, j + 1] = 0.125f;
                }
            }
        }

        /// <summary>
        /// Photoreceptor layer computation at each local cell
        /// </summary>
        /// <param name="pre_input"></param>
        /// <param name="cur_input"></param>
        /// <returns></returns>
        protected int pcellValue(byte pre_input, byte cur_input)
        {
            return cur_input - pre_input;
        }

        /// <summary>
        /// Spatiotemporal convolution forming inhibitions at each local cell
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="matrix"></param>
        /// <param name="kernel"></param>
        /// <param name="np"></param>
        /// <param name="pre_t"></param>
        /// <returns></returns>
        protected float icellValue(int x, int y, int[,,] matrix, float[,] kernel, int np, int pre_t)
        {
            float tmp = 0;
            int r, c;
            for (int i = -np; i < np + 1; i++)
            {
                r = x + i;
                while (r < 0)
                    r += 1;
                while (r >= height)
                    r -= 1;
                for (int j = -np; j < np + 1; j++)
                {
                    c = y + j;
                    while (c < 0)
                        c += 1;
                    while (c >= width)
                        c -= 1;
                    tmp += matrix[r, c, pre_t] * kernel[i + np, j + np];
                }
            }
            return tmp;
        }

        /// <summary>
        /// Summation layer computation at each local cell
        /// </summary>
        /// <param name="evalue"></param>
        /// <param name="ivalue"></param>
        /// <param name="wi"></param>
        /// <param name="ts"></param>
        /// <returns></returns>
        protected float scellValue(int evalue, float ivalue, float wi, int ts)
        {
            //return Math.Abs(pvalue) - Math.Abs(ivalue) * WI;
            if (evalue * ivalue <= 0)
                return 0;
            else
            {
                float tmpValue = evalue - ivalue * wi;
                if (tmpValue >= ts)
                    return tmpValue;
                else
                    return 0;
            }
        }

        /// <summary>
        /// Sigmoid transformation
        /// </summary>
        /// <param name="mp"></param>
        /// <param name="ncell"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        protected float Sigmoid(float mp, int ncell, float k)
        {
            return (float)Math.Pow(1 + Math.Exp(-mp * Math.Pow(ncell * k, -1)), -1);
        }

        /// <summary>
        /// Spiking mechanism
        /// </summary>
        /// <param name="smp"></param>
        /// <param name="tsp"></param>
        /// <param name="nsp"></param>
        /// <returns></returns>
        protected byte Spiking(float smp, float tsp, ref byte nsp)
        {
            if (smp >= tsp)
            {
                nsp += 1;
                return 1;
            }
            else
            {
                nsp = 0;
                return 0;
            }
        }

        /// <summary>
        /// Collision detection
        /// </summary>
        /// <param name="nsp"></param>
        /// <param name="nts"></param>
        /// <returns></returns>
        protected byte collisionDetecting(byte nsp, byte nts)
        {
            if (nsp >= nts)
                return 1;
            else
                return 0;
        }

        #endregion

        #region LGMD PROCESSING WITH A SINGLE PATHWAY

        /// <summary>
        /// Integrated visual processing of the LGMD1 mdoel with a single pathway
        /// </summary>
        /// <param name="img1"></param>
        /// <param name="img2"></param>
        /// <param name="t"></param>
        public void LGMDSProcessing(byte[,,] img1, byte[,,] img2, int t)
        {
            int cur_frame = t % 2;
            int pre_frame = (t - 1) % 2;
            float tmp_ffi = 0;
            mp = 0;
            //Photoreceptor and excitation layers processing
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Photoreceptors[y, x] = pcellValue(img1[y, x, 0], img2[y, x, 0]);
                    Excitations[y, x, cur_frame] = Photoreceptors[y, x];
                    tmp_ffi += Math.Abs(Excitations[y, x, pre_frame]);
                }
            }
            //FFI calculation and check
            ffi = tmp_ffi / Ncell;
            if (ffi >= Tffi)
            {
                mp = 0; //min value
            }
            else
            {
                //Inhibition and summation layers processing
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        Inhibitions[y, x] = icellValue(y, x, Excitations, WI, Np, pre_frame);
                        Summations[y, x] = scellValue(Excitations[y, x, cur_frame], Inhibitions[y, x], Wi, Ts);
                        mp += Summations[y, x];
                    }
                }
            }
            //LGMD1 membrane potential
            smp = Sigmoid(mp, Ncell, Csig);
            //Spiking
            spike = Spiking(smp, Tsp, ref Nsp);
            //Collision detecting
            if (collision == 0)
            {
                collision = collisionDetecting(Nsp, Nts);
                //attention
                if (collision == 1)
                {
                    activationTiming = t;
                }
            }

            //Print to Console
            Console.WriteLine("{0} {1:F} {2:F} {3:F} {4} {5}", t, mp, smp, ffi, spike, collision);
        }

        /// <summary>
        /// Exchange searching parameters to object domain (LGMD single pathway model)
        /// </summary>
        public void LGMDS_searchingParametersExchanging()
        {
            Tffi = (int)paramsDict["Tffi"];
            Ts = (int)paramsDict["Ts"];
            Tsp = paramsDict["Tsp"];
            Wi = paramsDict["Wi"];
            Csig = paramsDict["coe_sig"];
        }

        /// <summary>
        /// LGMD single pathway TNN model restore
        /// </summary>
        public void LGMDS_Restore()
        {
            Excitations = new int[height, width, 2];
        }

        /// <summary>
        /// Overwrite CompareTo method in IComparable
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public int CompareTo(Object obj)
        {
            if (this.fitness > ((LGMDSingle)obj).fitness)
                return -1;
            else if (this.fitness == ((LGMDSingle)obj).fitness)
                return 0;
            else
                return 1;
        }

        /// <summary>
        /// Overwrite Compare method in IComparer<LGMDSingle>
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public int Compare(LGMDSingle a, LGMDSingle b)
        {
            if (a.fitness > b.fitness)
                return 1;
            else if (a.fitness == b.fitness)
                return 0;
            else
                return -1;
        }

        #endregion
    }
}

