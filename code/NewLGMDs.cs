


/*
 * Filename: NewLGMD1.cs
 * Author: Qinbing FU
 * Location: Lincoln
 * Date: Feb 2019
 */


using System;


namespace LGMD
{
    /// <summary>
    /// Description:
    /// This is an updated modelling of LGMDs general visual neural network.
    /// This can implement both the LGMD1 and the LGMD2 neurons in locusts.
    /// Compared with previous works, this model has novel inhibition mechanisms adaptive to dynamic and cluttered backgrounds.
    /// The novelties of this model are as follows:
    /// 1.  modelling of trans-medullary-afferents (TmAs): self and lateral inhibition mechanisms in the ON and OFF pathways (all local inhibitions are temporally delayed relative to local excitations.).
    /// 2.  a feed forward inhibition mediation pathway mediates the local biases in the dual pathways.
    /// 3.  temporal dynamics: latencies of local excitations are shortened along with acceleration of the image edge motion as collision nears.
    /// 4.  spike frequency calculation (spikes per second)
    /// 5.  Adjusting the local weightings and the temporal delays and the coefficients of supralinear interaction between ON and OFF local excitations can realise either the LGMD1 or the LGMD2.
    /// 
    /// References: [1] Q. Fu, H. Wang, S. Yue, "Developed Visual System for Robust Collision Recognition in Various Automotive Scenes", 2019.
    ///             [2] Q. Fu, C. Hu, J. Peng, F. C. Rind, S. Yue, "A Robust Collision Perception Visual Neural Network with Specific Selectivity to Darker Objects", IEEE Transactions on Cybernetics, 2019.
    ///             [3] Q. Fu, C. Hu, J. Peng, S. Yue, "Shaping the Collision Selectivity in a Looming Sensitive Neuron Model with Parallel ON and OFF Pathways and Spike Frequency Adaptation", Neural Networks, 2018.
    ///             [4] Q. Fu, H. Wang, C. Hu, S. Yue, "Towards Computational Models and Applications of Insect Visual Systems for Motion Perception: A Review", Artificial Life, 2019.
    ///             [5] Q. Fu, C. Hu, T. Liu, S. Yue, "Collision Selective LGMDs Neuron Models Research Benefits from a Vision-based Autonomous Micro Robot", IROS, 2017.
    ///             [6] Q. Fu, S. Yue, C. Hu, "Bio-inspired Collision Detector with Enhanced Selectivity for Ground Robotic Vision System", BMVC, 2016.
    ///             [7] Q. Fu, N. Bellotto, H. Wang, F. C. Rind, H. Wang, S. Yue, "A Visual Neural Network for Robust Collision Perception in Vehicle Driving Scenarios", AIAI, 2019.
    /// </summary>
    internal class NewLGMDs : LGMDs
    {
        #region FIELDS

        protected float[,] Conv_High;   //convolution matrix with high weights
        protected float[,] Conv_Low;  //convolution matrix with low weights
        protected float[,] Inh_ON;    //delayed inhibitions in ON channels
        protected float[,] Inh_OFF;   //delayed inhibitions in OFF channels
        protected float[] tau_Small;     //small time constants in milliseconds in ON and OFF channels
        protected float[] tau_Large;    //large time constants in milliseconds in ON and OFF channels
        protected float tau_E;        //time delay of local excitation after grouping layer in milliseconds
        protected float raw_tau_E;    //raw time delay of local excitation
        protected float tau_FFI;      //time delay of FFI in milliseconds
        protected float[] lp_Small;      //small delay coefficient in ON/OFF pathways
        protected float[] lp_Large;     //large delay coefficient in ON/OFF pathways
        protected float lp_FFI;       //delay coefficient in FFI pathway
        protected float lp_E;         //delay coefficient in local excitatory input
        protected float raw_lp_E;     //raw delay coefficient in local excitatory input
        protected int Cw;             //a constant to compute the scale in G-layer
        protected float Delta_C;      //a small real number to compute the scale in G-layer
        protected float Cde;          //local excitation decay coefficient
        protected int Tde;            //local excitation decay thresholding
        protected float W_i_high;       //high local bias for inhibitions
        protected float W_i_low;        //low local bias for inhibitions
        protected float dyn_bias;     //dynamic inhibition bias in ON and OFF channels
        protected new float[,,] gcells; //new grouping layer
        protected float spiRate;  //spike rate
        protected new byte[] spike;   //spikes

        #endregion

        #region PROPERTIES

        public float[,] INH_ON
        {
            get { return Inh_ON; }
            set { Inh_ON = value; }
        }

        public float[,] INH_OFF
        {
            get { return Inh_OFF; }
            set { Inh_OFF = value; }
        }

        public float TAU_EXC
        {
            get { return tau_E; }
            set { tau_E = value; }
        }

        public float DYNAMIC_BIAS
        {
            get { return dyn_bias; }
            set { dyn_bias = value; }
        }

        public new float[,,] G_CELLS
        {
            get { return gcells; }
            set { gcells = value; }
        }

        public float SPIKERATE
        {
            get { return spiRate; }
            set { spiRate = value; }
        }

        public new byte[] SPIKE
        {
            get { return spike; }
            set { spike = value; }
        }

        #endregion

        #region CONSTRUCTORS

        /// <summary>
        /// Default Constructor
        /// </summary>
        public NewLGMDs() { }

        /// <summary>
        /// Parameterised Constructor
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="fps"></param>
        public NewLGMDs(int width, int height, int fps) : base(width, height, fps)
        {
            this.Conv_High = new float[2 * Np + 1, 2 * Np + 1];
            this.Conv_Low = new float[2 * Np + 1, 2 * Np + 1];
            this.Conv_High = makeHighConvKernel();
            this.Conv_Low = makeLowConvKernel();
            this.Inh_ON = new float[height, width];
            this.Inh_OFF = new float[height, width];
            this.tau_Small = new float[2 * Np + 1];  //[0] for centre, [1] for nearest, [2] for diagonal
            this.tau_Large = new float[2 * Np + 1];  //[0] for centre, [1] for nearest, [2] for diagonal
            this.lp_Small = new float[2 * Np + 1];   //[0] for centre, [1] for nearest, [2] for diagonal
            this.lp_Large = new float[2 * Np + 1];   //[0] for centre, [1] for nearest, [2] for diagonal
            for (int i = 0; i < 2 * Np + 1; i++)
            {
                this.tau_Small[i] = 15 + i * 15;    //ms
                this.tau_Large[i] = 60 + i * 60;   //ms
                this.lp_Small[i] = time_interval / (time_interval + tau_Small[i]);
                this.lp_Large[i] = time_interval / (time_interval + tau_Large[i]);
            }
            this.tau_FFI = 10;   //ms
            this.lp_FFI = time_interval / (time_interval + tau_FFI);
            this.raw_tau_E = 10;   //ms
            this.raw_lp_E = time_interval / (time_interval + raw_tau_E);
            this.tau_E = this.raw_tau_E;
            this.lp_E = this.raw_lp_E;
            this.W_i_high = 1;
            this.W_i_low = 0.5f;
            this.W_on = 1;
            this.W_off = 1;
            this.W_onoff = 0;
            this.Cw = 4;
            this.Delta_C = 0.5f;
            this.Cde = 0.5f;
            this.Tde = 15;
            this.gcells = new float[height, width, 2];
            this.spiRate = 0;
            this.Nts = 6;
            //this.Nts = (int)(1000 / time_interval);
            this.spike = new byte[Nts];
            this.dyn_bias = W_i_low;

            //base parameters override initialization
            this.coe_sig = 1;
            this.clip_point = 0;
            this.Tffi = 20;
            //this.Tsf = 0.003f;
            //this.Tsp = 0.78f;
            this.Ksp = 10;
            this.tau_hp = 500;
            this.hp_delay = this.tau_hp / (base.time_interval + this.tau_hp);
            
            Console.WriteLine("New LGMDs Visual Neural Network Parameters Set-up Ready......\n");
        }

        #endregion

        #region ALGORITHMS AND FUNCTIONS

        /// <summary>
        /// Making a convolution kernel in ON pathway
        /// </summary>
        /// <returns></returns>
        private float[,] makeHighConvKernel()
        {
            float[,] mat = new float[2 * Np + 1, 2 * Np + 1];
            for (int i = -1; i < Np + 1; i++)
            {
                for (int j = -1; j < Np + 1; j++)
                {
                    if (i == 0 && j == 0)   //centre
                        mat[i + 1, j + 1] = 2;
                    else if (i == 0 || j == 0)  //nearest
                        mat[i + 1, j + 1] = 0.5f;
                    else   //diagonal
                        mat[i + 1, j + 1] = 0.25f;
                }
            }
            return mat;
        }

        /// <summary>
        /// Making a convolution kernel in OFF pathway
        /// </summary>
        /// <returns></returns>
        private float[,] makeLowConvKernel()
        {
            float[,] mat = new float[2 * Np + 1, 2 * Np + 1];
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
            return mat;
        }

        /// <summary>
        /// FFI-M pathway adjusts the local bias in the dual-channels
        /// </summary>
        /// <param name="cur_ffi"></param>
        /// <param name="init_w"></param>
        /// <returns></returns>
        protected float FFIMediation(float cur_ffi, float init_w)
        {
            float bias = cur_ffi / Tffi;
            if (bias <= init_w)
                return init_w;
            else
                return bias;
        }

        /// <summary>
        /// FFI shapes delay coefficient in ON and OFF pathways
        /// </summary>
        /// <param name="cur_ffi"></param>
        /// <returns></returns>
        protected float FFIshapesDelayCoefficient(float cur_ffi)
        {
            float tmp = 1 - cur_ffi / Tffi;
            if (tmp < 0)
                return Delta_C; //return a very small real number
            else
                return tmp;
        }

        /// <summary>
        /// Adaptive temporal tuning
        /// </summary>
        /// <param name="raw_tau"></param>
        /// <param name="tau"></param>
        /// <param name="delay"></param>
        /// <param name="sigma"></param>
        protected void temporalTuning(float raw_tau, ref float tau, ref float delay, float sigma)
        {
            tau = raw_tau * sigma;
            delay = time_interval / (time_interval + tau);
        }

        /// <summary>
        /// Spatiotemporal convolution in ON and OFF pathways
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="inputMatrix"></param>
        /// <param name="kernel"></param>
        /// <param name="cur_frame"></param>
        /// <param name="pre_frame"></param>
        /// <param name="lp_delay"></param>
        /// <returns></returns>
        protected float Convolution(int x, int y, float[,,] inputMatrix, float[,] kernel, int cur_frame, int pre_frame, float[] lp_delay)
        {
            float tmp = 0;
            int r, c;
            float lp;
            for (int i = -Np; i < Np + 1; i++)
            {
                //check border
                r = x + i;
                while (r < 0)
                    r += 1;
                while (r >= height)
                    r -= 1;
                for (int j = -Np; j < Np + 1; j++)
                {
                    //check border
                    c = y + j;
                    while (c < 0)
                        c += 1;
                    while (c >= width)
                        c -= 1;
                    //centre cell
                    if (i == 0 && j == 0)
                        lp = LowpassFilter(inputMatrix[r, c, cur_frame], inputMatrix[r, c, pre_frame], lp_delay[0]);
                    //nearest cells
                    else if (i == 0 || j == 0)
                        lp = LowpassFilter(inputMatrix[r, c, cur_frame], inputMatrix[r, c, pre_frame], lp_delay[1]);
                    //diagonal cells
                    else
                        lp = LowpassFilter(inputMatrix[r, c, cur_frame], inputMatrix[r, c, pre_frame], lp_delay[2]);
                    tmp += lp * kernel[i + Np, j + Np];
                }
            }
            return tmp;
        }

        /// <summary>
        /// A scale computation in G layer
        /// </summary>
        /// <param name="cur_frame"></param>
        /// <returns></returns>
        protected float Scale(int cur_frame)
        {
            float max = 0;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    if (max < Math.Abs(gcells[i, j, cur_frame]))
                        max = Math.Abs(gcells[i, j, cur_frame]);
                }
            }
            return (float)(Delta_C + max * Math.Pow(Cw, -1));
        }

        /// <summary>
        /// Computation and thresholding of each Grouping cell
        /// </summary>
        /// <param name="scellvalue"></param>
        /// <param name="ce"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        protected float gCellValue(float scellvalue, float ce, float w)
        {
            float value = scellvalue * ce * (float)Math.Pow(w, -1);
            if (value * Cde >= Tde)
                return value;
            else
                return 0;
        }

        /// <summary>
        /// Override summation method in the ON and OFF pathways
        /// </summary>
        /// <param name="on_exc"></param>
        /// <param name="off_exc"></param>
        /// <returns></returns>
        protected new float SupralinearSummation(float on_exc, float off_exc)
        {
            return W_on * on_exc + W_off * off_exc + W_onoff * on_exc * off_exc;
        }

        /// <summary>
        /// Computation of each Summation cell
        /// </summary>
        /// <param name="exc"></param>
        /// <param name="inh"></param>
        /// <param name="wi"></param>
        /// <returns></returns>
        protected float sCellValue(float exc, float inh, float wi)
        {
            float tmp = exc - inh * wi;
            if (tmp <= 0)
                return 0;
            else
                return tmp;
        }

        /// <summary>
        /// Calculation of spike frequency
        /// </summary>
        /// <param name="spike"></param>
        /// <returns></returns>
        protected float spikeFrequency(byte[] spike)
        {
            int spikeCount = 0;
            for (int i = 0; i < spike.Length; i++)
            {
                spikeCount += spike[i];
            }
            return spikeCount * 1000 / (spike.Length * base.time_interval);
        }

        #endregion

        #region LGMDs VISUAL NEURAL NETWORK PROCESSING

        /// <summary>
        /// LGMD2 based visual processing
        /// </summary>
        /// <param name="img1"></param>
        /// <param name="img2"></param>
        /// <param name="t"></param>
        public void LGMD_Processing(byte[,,] img1, byte[,,] img2, int t)
        {
            float tmp_sum = 0;
            int cur_frame = t % ONs.GetLength(2);
            int pre_frame = (t - 1) % ONs.GetLength(2);
            int cur_spi = t % SPIKE.Length;
            float tmp_ffi = 0;
            float sigma, sca;

            //PHOTORECEPTORS
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    PHOTOS[y, x, cur_frame] = HighpassFilter(img1[y, x, 0], img2[y, x, 0]);
                    tmp_ffi += Math.Abs(PHOTOS[y, x, cur_frame]);
                }
            }

            /*attention!*/
            //FFI TUNING
            FFI[cur_frame] = tmp_ffi / Ncell;
            FFI[cur_frame] = LowpassFilter(FFI[cur_frame], FFI[pre_frame], lp_FFI);
            sigma = FFIshapesDelayCoefficient(FFI[cur_frame]);
            temporalTuning(raw_tau_E, ref tau_E, ref lp_E, sigma);
            this.dyn_bias = FFIMediation(FFI[cur_frame], W_i_low);
            //dyn_bias = W_i_low;
            /*the new tuning mechanism*/

            //ON AND OFF RECTIFYING
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    ONs[y, x, cur_frame] = HRplusDC_ON(ONs[y, x, pre_frame], PHOTOS[y, x, cur_frame]);
                    OFFs[y, x, cur_frame] = HRplusDC_OFF(OFFs[y, x, pre_frame], PHOTOS[y, x, cur_frame]);
                }
            }

            //SPATIOTEMPORAL PROCESSING IN DUAL-PATHWAYS
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    INH_ON[y, x] = Convolution(y, x, ONs, Conv_Low, cur_frame, pre_frame, lp_Large);
                    INH_OFF[y, x] = Convolution(y, x, OFFs, Conv_Low, cur_frame, pre_frame, lp_Large);
                    S_on = sCellValue(ONs[y, x, cur_frame], INH_ON[y, x], dyn_bias);
                    S_off = sCellValue(OFFs[y, x, cur_frame], INH_OFF[y, x], dyn_bias);
                    S_CELLS[y, x] = SupralinearSummation(S_on, S_off);
                }
            }
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    G_CELLS[y, x, cur_frame] = Convolving(y, x, S_CELLS, W_g);
                }
            }
            sca = Scale(cur_frame);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    G_CELLS[y, x, cur_frame] = gCellValue(S_CELLS[y, x], G_CELLS[y, x, cur_frame], sca);
                    //latency of input local excitation
                    G_CELLS[y, x, cur_frame] = LowpassFilter(G_CELLS[y, x, cur_frame], G_CELLS[y, x, pre_frame], lp_E);
                    tmp_sum += G_CELLS[y, x, cur_frame];
                }
            }

            //MEMBRANE POTENTIAL
            MP[cur_frame] = tmp_sum;
            SMP[cur_frame] = SigmoidTransfer(MP[cur_frame]);

            //SPIKE FREQUENCY ADAPTATION
            SFA[cur_frame] = SFA_HPF(SFA[pre_frame], SMP[pre_frame], SMP[cur_frame]);

            //SPIKING
            SPIKE[cur_spi] = Spiking(SFA[cur_frame]);

            //RATE
            SPIKERATE = spikeFrequency(SPIKE);

            Console.WriteLine("{0} {1:F} {2:F} {3:F} {4:F} {5} {6:F} {7:F} {8:F}", t, this.MP[cur_frame], this.SMP[cur_frame], this.SFA[cur_frame], this.FFI[cur_frame], this.SPIKE[cur_spi], this.SPIKERATE, this.DYNAMIC_BIAS, this.TAU_EXC);
        }

        #endregion
    }
}


