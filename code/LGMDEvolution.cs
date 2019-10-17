/*
 * Filename: LGMDEvolution.cs
 * Author: Qinbing FU
 * Location: Guangzhou
 * Date: May-September 2019
 */


using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;


namespace LGMD
{
    /// <summary>
    /// Genetic algorithms training data structure used in LGMD models competitive coevolution
    /// References: [1] Q. Fu, H. Wang, S. Yue, "Developed Visual System for Robust Collision Recognition in Various Automotive Scenes", 2019.
    ///             [2] S. Yue, F. C. Rind, "Redundant Neural Vision Systems - Competing for collision Recognition Roles", IEEE Transactions on Autonomous Mental Development, 2013.
    /// </summary>
    internal struct GAsData
    {
        /// <summary>
        /// data ID
        /// </summary>
        public int ID;
        /// <summary>
        /// whether a collision event (1) or not (0)
        /// </summary>
        public int collision;
        /// <summary>
        /// start frame point of labelled crash event
        /// </summary>
        public int eventStartPoint;
        /// <summary>
        /// end frame point of labelled crash event
        /// </summary>
        public int eventEndPoint;
        /// <summary>
        /// allowed error to event labelling (in frames)
        /// </summary>
        public int allowed_error;
        /// <summary>
        /// data file name
        /// </summary>
        public string fileName;
    }

    /// <summary>
    /// This class implements evolutionary computation of various LGMD models.
    /// A genetic algorithm is used in this class.
    /// </summary>
    internal sealed class LGMDEvolution
    {
        #region FIELD

        /// <summary>
        /// random number generator
        /// </summary>
        private static Random _rand = new Random(GetRandomSeed());
        /// <summary>
        /// video data capture from file
        /// </summary>
        private Capture GACapture;
        /// <summary>
        /// frame width
        /// </summary>
        private int width;
        /// <summary>
        /// frame height
        /// </summary>
        private int height;
        /// <summary>
        /// video data frames per second
        /// </summary>
        private int fps;
        /// <summary>
        /// total frame number
        /// </summary>
        private int frames;
        /// <summary>
        /// byte array to store gray-scale image
        /// </summary>
        private byte[,,] img1;
        /// <summary>
        /// byte array to store gray-scale image
        /// </summary>
        private byte[,,] img2;
        /// <summary>
        /// image matrices list
        /// </summary>
        private List<byte[,,]> photos;
        /// <summary>
        /// image frames
        /// </summary>
        private Image<Gray, Byte>[] images;
        /// <summary>
        /// GAs dataset
        /// </summary>
        private GAsData[] dataset;
        /// <summary>
        /// LGMD TNN model (with a single processing pathway) objects
        /// </summary>
        private List<LGMDSingle> objLGMDS;
        /// <summary>
        /// LGMD NN model (with ON and OFF dual processing pathways) objects
        /// </summary>
        private List<LGMDs> objLGMDD;
        /// <summary>
        /// LGMDPlus model objects
        /// </summary>
        private List<LGMDPlus> objLGMDPlus;
        /// <summary>
        /// initial LGMD TNN model agents number
        /// </summary>
        private int initLGMDSAgentsNo;
        /// <summary>
        /// initial LGMD NN model agents number
        /// </summary>
        private int initLGMDDAgentsNo;
        /// <summary>
        /// initial LGMDPlus model agents number
        /// </summary>
        private int initLGMDPAgentsNo;
        /// <summary>
        /// initial total agents
        /// </summary>
        private int totalAgents;
        /// <summary>
        /// LGMD TNN model agents number in GA process
        /// </summary>
        private int processLGMDSAgentsNo;
        /// <summary>
        /// LGMD NN model agents number in GA process
        /// </summary>
        private int processLGMDDAgentsNo;
        /// <summary>
        /// LGMDPlus model agents number in GA process
        /// </summary>
        private int processLGMDPAgentsNo;
        /// <summary>
        /// crossover probability
        /// </summary>
        private readonly double Pc;
        /// <summary>
        /// global mutation probability
        /// </summary>
        private readonly double Pm;
        /// <summary>
        /// crash events number
        /// </summary>
        private int crashEventsNO;
        /// <summary>
        /// dataset label txt file
        /// </summary>
        private string labelTxtFile;
        /// <summary>
        /// output txt file
        /// </summary>
        private string outputTxtFile;
        /// <summary>
        /// standard deviation in Gaussian Perturbation
        /// </summary>
        private readonly float sigma;
        /// <summary>
        /// scale parameter in Gaussian Perturbation
        /// </summary>
        private readonly float gauss_scale;
        /// <summary>
        /// maximum likelihood of Gaussian distribution probability density function
        /// </summary>
        private float max_likelihood;
        /// <summary>
        /// minimum likelihood of Gaussian distribution probability density function
        /// </summary>
        private float min_likelihood;
        /// <summary>
        /// maximum distance to mean value of Gaussian distribution
        /// </summary>
        private readonly float max_distanceToMean;
        /// <summary>
        /// loss of collision miss-detection
        /// </summary>
        private readonly float lossCol;
        /// <summary>
        /// loss of error detection
        /// </summary>
        private readonly float lossNon;
        /// <summary>
        /// score of failure fitness covering all events
        /// </summary>
        private readonly float fitnessFScore;
        /// <summary>
        /// collision events amount in training data
        /// </summary>
        private readonly int Ncol;
        /// <summary>
        /// non-collision events amount in training data
        /// </summary>
        private readonly int Nnon;
        /// <summary>
        /// number of parent agents to select from population, which should be an even number
        /// </summary>
        private readonly int Npa;
        /// <summary>
        /// GAs generation
        /// </summary>
        private int generation;
        /// <summary>
        /// max GAs generation
        /// </summary>
        private readonly int maxGeneration;
        /// <summary>
        /// mean fitness value: 0->LGMDPlus 1-> LGMDS 2->LGMDD
        /// </summary>
        private float[] meanFitness;
        /// <summary>
        /// max fitness value: 0->LGMDPlus 1->LGMDS 2->LGMDD
        /// </summary>
        private float[] maxFitness;
        /// <summary>
        /// sum of fitness value: 0->LGMDPlus 1->LGMDS 2->LGMDD
        /// </summary>
        private float[] sumFitness;
        /// <summary>
        /// number of best agents: 0->LGMDPlus 1->LGMDS 2->LGMDD
        /// </summary>
        private int[] bestAgents;
        /// <summary>
        /// fitness baseline determining best agents
        /// </summary>
        private float fitBaseline;
        /// <summary>
        /// GA kind: 1->LGMDPlus 2->LGMDS 3->LGMDD 4->coevolution
        /// </summary>
        private int kindGA;

        #endregion

        #region PROPERTY

        /// <summary>
        /// property of training dataset
        /// </summary>
        public GAsData[] Dataset
        {
            get { return dataset; }
        }

        /// <summary>
        /// property of LGMD TNN model objects
        /// </summary>
        public List<LGMDSingle> ObjLGMDS
        {
            get { return objLGMDS; }
            //set { objLGMDS = value; }
        }

        /// <summary>
        /// property of LGMD NN model objects
        /// </summary>
        public List<LGMDs> ObjLGMDD
        {
            get { return objLGMDD; }
            //set { objLGMDD = value; }
        }

        /// <summary>
        /// property of LGMDPlus model objects
        /// </summary>
        public List<LGMDPlus> ObjLGMDP
        {
            get { return objLGMDPlus; }
            //set { objLGMDPlus = value; }
        }

        /// <summary>
        /// property of process LGMD TNN agents number in GA
        /// </summary>
        public int ProcessLGMDSNo
        {
            get { return processLGMDSAgentsNo; }
            //set { processLGMDSAgentsNo = value; }
        }

        /// <summary>
        /// property of process LGMD NN agents number in GA
        /// </summary>
        public int ProcessLGMDDNo
        {
            get { return processLGMDDAgentsNo; }
            //set { processLGMDDAgentsNo = value; }
        }

        /// <summary>
        /// property of process LGMDPlus agents number in GA
        /// </summary>
        public int ProcessLGMDPNo
        {
            get { return processLGMDPAgentsNo; }
            //set { processLGMDPAgentsNo = value; }
        }

        /// <summary>
        /// property of GA generation
        /// </summary>
        public int GAsGeneration
        {
            get { return generation; }
            //set { generation = value; }
        }

        /// <summary>
        /// property of max fitness value
        /// </summary>
        public float[] MaxFitness
        {
            get { return maxFitness; }
            //set { maxFitness = value; }
        }

        /// <summary>
        /// property of mean fitness value
        /// </summary>
        public float[] MeanFitness
        {
            get { return meanFitness; }
            //set { meanFitness = value; }
        }

        /// <summary>
        /// property of sum of fitness value
        /// </summary>
        public float[] SumFitness
        {
            get { return sumFitness; }
            //set { sumFitness = value; }
        }

        /// <summary>
        /// property of best agents
        /// </summary>
        public int[] BestAgents
        {
            get { return bestAgents; }
            //set { bestAgents = value; }
        }

        /// <summary>
        /// property of frame width
        /// </summary>
        public int Width
        {
            get { return width; }
        }

        /// <summary>
        /// property of frame height
        /// </summary>
        public int Height
        {
            get { return height; }
        }

        /// <summary>
        /// property of frames per second
        /// </summary>
        public int Fps
        {
            get { return fps; }
        }

        #endregion

        #region CONSTRUCTOR

        /// <summary>
        /// Constructor
        /// </summary>
        public LGMDEvolution() { }

        /// <summary>
        /// Parameterised constructor
        /// </summary>
        /// <param name="initLGMDSAgentsNo"></param>
        /// <param name="initLGMDDAgentsNo"></param>
        /// <param name="initLGMDPAgentsNo"></param>
        /// <param name="labelTxtFile"></param>
        /// <param name="outputTxtFile"></param>
        /// <param name="kindGA"></param>
        public LGMDEvolution(int initLGMDSAgentsNo, int initLGMDDAgentsNo, int initLGMDPAgentsNo, string labelTxtFile, /*string outputTxtFile,*/ int kindGA)
        {
            //attention
            width = 426;
            height = 240;
            fps = 30;
            Ncol = 51;
            Nnon = 36;
            //Ncol = 3;
            //Nnon = 1;
            Npa = 8; //shoule be an even number, 40% of initial population for parents, 20% of initial population for offsprings
            dataset = new GAsData[Ncol + Nnon]; //fixed number of training data in GAs pool
            this.initLGMDSAgentsNo = initLGMDSAgentsNo;
            this.initLGMDDAgentsNo = initLGMDDAgentsNo;
            this.initLGMDPAgentsNo = initLGMDPAgentsNo;
            this.labelTxtFile = labelTxtFile;
            //this.outputTxtFile = outputTxtFile;
            outputTxtFile = "ga";
            this.kindGA = kindGA;
            objLGMDPlus = new List<LGMDPlus>();
            objLGMDS = new List<LGMDSingle>();
            objLGMDD = new List<LGMDs>();
            Pc = 0.9;
            Pm = 0.3;
            crashEventsNO = loadEventPoints(labelTxtFile);
            sigma = 1;
            max_distanceToMean = 9;
            gauss_scale = 3 * sigma;
            min_likelihood = gaussianDensityCalc(sigma, max_distanceToMean);
            max_likelihood = gaussianDensityCalc(sigma, 0);
            lossCol = 3;
            lossNon = 1;
            fitnessFScore = lossCol * Ncol + lossNon * Nnon;    // 100 failure score in total
            generation = 1;
            maxGeneration = 50;
            processLGMDSAgentsNo = initLGMDSAgentsNo;
            processLGMDDAgentsNo = initLGMDDAgentsNo;
            processLGMDPAgentsNo = initLGMDPAgentsNo;
            totalAgents = initLGMDSAgentsNo + initLGMDDAgentsNo + initLGMDPAgentsNo;
            fitBaseline = 80;
            meanFitness = new float[3];
            maxFitness = new float[3];
            sumFitness = new float[3];
            bestAgents = new int[3];

            Console.WriteLine("GAs of LGMD models environment initialised.....");
        }

        #endregion

        #region METHOD

        /// <summary>
        /// Load event text file
        /// </summary>
        /// <param name="labelTxtFile"></param>
        /// <returns></returns>
        private int loadEventPoints(string labelTxtFile)
        {
            int num = 1;
            int dummy;
            using (StreamReader rf = new StreamReader(labelTxtFile))
            {
                if (rf == null)
                {
                    Console.WriteLine("Label file not found...\n");
                    return 0;
                }
                string s;
                while ((s = rf.ReadLine()) != null)
                {
                    Console.WriteLine(s);
                    //Split each line by 'space'
                    string[] split = s.Split(' ');
                    dummy = int.Parse(split[0]);
                    if (dummy != num)
                    {
                        Console.WriteLine("Label file corrupted...\n");
                        return 0;
                    }
                    dataset[dummy - 1].ID = num;
                    dataset[dummy - 1].eventStartPoint = int.Parse(split[1]);
                    dataset[dummy - 1].eventEndPoint = int.Parse(split[2]);
                    dataset[dummy - 1].allowed_error = fps;
                    dataset[dummy - 1].fileName = split[3];
                    dataset[dummy - 1].collision = int.Parse(split[4]);
                    num++;
                }
            }
            return num;
        }

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
        /// Gaussian density calculation
        /// </summary>
        /// <param name="sigma"></param>
        /// <param name="distanceToMean"></param>
        /// <returns></returns>
        private float gaussianDensityCalc(float sigma, float distanceToMean)
        {
            return (float)(Math.Exp((0 - distanceToMean) / (2 * sigma * sigma)) / Math.Sqrt(2 * Math.PI * sigma * sigma));
        }

        /// <summary>
        /// Calculation of distance to mean in gaussian density function
        /// </summary>
        /// <param name="density"></param>
        /// <param name="sigma"></param>
        /// <returns></returns>
        private double gaussianDistanceCalc(double density, float sigma)
        {
            return -2 * sigma * sigma * Math.Log(density * Math.Sqrt(2 * Math.PI * sigma * sigma));
        }

        /// <summary>
        /// Initialisation of population
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="fps"></param>
        private void agentsInitialisation(int width, int height, int fps)
        {
            switch (kindGA)
            {
                case 1:
                    {
                        LGMDPlus[] lgmdP = new LGMDPlus[initLGMDPAgentsNo];
                        lgmdP[0] = new LGMDPlus(width, height, fps);
                        objLGMDPlus.Add(lgmdP[0]);
                        for (int i = 1; i < initLGMDPAgentsNo; i++)
                        {
                            lgmdP[i] = new LGMDPlus(width, height, fps);
                            lgmdP[i].ParamsDict["Tffi"] = gaussPerturbation(lgmdP[i].ParamsDict["Tffi"], lgmdP[i].min_Tffi, lgmdP[i].max_Tffi, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["Tde"] = gaussPerturbation(lgmdP[i].ParamsDict["Tde"], lgmdP[i].min_Tde, lgmdP[i].max_Tde, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["Tsp"] = gaussPerturbation(lgmdP[i].ParamsDict["Tsp"], lgmdP[i].min_Tsp, lgmdP[i].max_Tsp, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["Tsr"] = gaussPerturbation(lgmdP[i].ParamsDict["Tsr"], lgmdP[i].min_Tsr, lgmdP[i].max_Tsr, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["tau_E"] = gaussPerturbation(lgmdP[i].ParamsDict["tau_E"], lgmdP[i].min_tau_cen_E, lgmdP[i].max_tau_cen_E, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["tau_hp"] = gaussPerturbation(lgmdP[i].ParamsDict["tau_hp"], lgmdP[i].min_tau_hp, lgmdP[i].max_tau_hp, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["W_base"] = gaussPerturbation(lgmdP[i].ParamsDict["W_base"], lgmdP[i].min_W_base, lgmdP[i].max_W_base, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["coe_sig"] = gaussPerturbation(lgmdP[i].ParamsDict["coe_sig"], lgmdP[i].min_Csig, lgmdP[i].max_Csig, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["std_w"] = gaussPerturbation(lgmdP[i].ParamsDict["std_w"], lgmdP[i].min_std_w, lgmdP[i].max_std_w, sigma, gauss_scale);
                            lgmdP[i].LGMDPlus_searchingParametersExchanging();
                            objLGMDPlus.Add(lgmdP[i]);
                        }
                        Console.WriteLine("GAs population initialisation: LGMDPlus");
                        break;
                    }
                case 2:
                    {
                        LGMDSingle[] lgmdS = new LGMDSingle[initLGMDSAgentsNo];
                        lgmdS[0] = new LGMDSingle(width, height);
                        objLGMDS.Add(lgmdS[0]);
                        for (int i = 1; i < initLGMDSAgentsNo; i++)
                        {
                            lgmdS[i] = new LGMDSingle(width, height);
                            lgmdS[i].ParamsDict["Tffi"] = gaussPerturbation(lgmdS[i].ParamsDict["Tffi"], lgmdS[i].min_Tffi, lgmdS[i].max_Tffi, sigma, gauss_scale);
                            lgmdS[i].ParamsDict["Ts"] = gaussPerturbation(lgmdS[i].ParamsDict["Ts"], lgmdS[i].min_Ts, lgmdS[i].max_Ts, sigma, gauss_scale);
                            lgmdS[i].ParamsDict["Tsp"] = gaussPerturbation(lgmdS[i].ParamsDict["Tsp"], lgmdS[i].min_Tsp, lgmdS[i].max_Tsp, sigma, gauss_scale);
                            lgmdS[i].ParamsDict["Wi"] = gaussPerturbation(lgmdS[i].ParamsDict["Wi"], lgmdS[i].min_Wi, lgmdS[i].max_Wi, sigma, gauss_scale);
                            lgmdS[i].ParamsDict["coe_sig"] = gaussPerturbation(lgmdS[i].ParamsDict["coe_sig"], lgmdS[i].min_Csig, lgmdS[i].max_Csig, sigma, gauss_scale);
                            lgmdS[i].LGMDS_searchingParametersExchanging();
                            objLGMDS.Add(lgmdS[i]);
                        }
                        Console.WriteLine("GAs population initialisation: LGMD TNN");
                        break;
                    }
                case 3:
                    {
                        LGMDs[] lgmdD = new LGMDs[initLGMDDAgentsNo];
                        lgmdD[0] = new LGMDs(width, height, fps);
                        objLGMDD.Add(lgmdD[0]);
                        for (int i = 1; i < initLGMDDAgentsNo; i++)
                        {
                            lgmdD[i] = new LGMDs(width, height, fps);
                            lgmdD[i].ParamsDict["Tffi"] = gaussPerturbation(lgmdD[i].ParamsDict["Tffi"], lgmdD[i].min_Tffi, lgmdD[i].max_Tffi, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["Ts"] = gaussPerturbation(lgmdD[i].ParamsDict["Ts"], lgmdD[i].min_Ts, lgmdD[i].max_Ts, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["Tsp"] =  gaussPerturbation(lgmdD[i].ParamsDict["Tsp"], lgmdD[i].min_Tsp, lgmdD[i].max_Tsp, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["coe_sig"] = gaussPerturbation(lgmdD[i].ParamsDict["coe_sig"], lgmdD[i].min_Csig, lgmdD[i].max_Csig, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["W_i_off"] = gaussPerturbation(lgmdD[i].ParamsDict["W_i_off"], lgmdD[i].min_W_i_off, lgmdD[i].max_W_i_off, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["W_i_on"] = gaussPerturbation(lgmdD[i].ParamsDict["W_i_on"], lgmdD[i].min_W_i_on, lgmdD[i].max_W_i_on, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["tau_hp"] = gaussPerturbation(lgmdD[i].ParamsDict["tau_hp"], lgmdD[i].min_tau_hp, lgmdD[i].max_tau_hp, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["tau_lp"] = gaussPerturbation(lgmdD[i].ParamsDict["tau_lp"], lgmdD[i].min_tau_lp, lgmdD[i].max_tau_lp, sigma, gauss_scale);
                            lgmdD[i].LGMDD_searchingParametersExchanging();
                            objLGMDD.Add(lgmdD[i]);
                        }
                        Console.WriteLine("GAs population initialisation: LGMG NN");
                        break;
                    }
                case 4:
                    {
                        LGMDPlus[] lgmdP = new LGMDPlus[initLGMDPAgentsNo];
                        LGMDs[] lgmdD = new LGMDs[initLGMDDAgentsNo];
                        LGMDSingle[] lgmdS = new LGMDSingle[initLGMDSAgentsNo];
                        lgmdP[0] = new LGMDPlus(width, height, fps);
                        objLGMDPlus.Add(lgmdP[0]);
                        lgmdD[0] = new LGMDs(width, height, fps);
                        objLGMDD.Add(lgmdD[0]);
                        lgmdS[0] = new LGMDSingle(width, height);
                        objLGMDS.Add(lgmdS[0]);
                        for (int i = 1; i < initLGMDPAgentsNo; i++)
                        {
                            lgmdP[i] = new LGMDPlus(width, height, fps);
                            lgmdP[i].ParamsDict["Tffi"] = gaussPerturbation(lgmdP[i].ParamsDict["Tffi"], lgmdP[i].min_Tffi, lgmdP[i].max_Tffi, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["Tde"] = gaussPerturbation(lgmdP[i].ParamsDict["Tde"], lgmdP[i].min_Tde, lgmdP[i].max_Tde, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["Tsp"] = gaussPerturbation(lgmdP[i].ParamsDict["Tsp"], lgmdP[i].min_Tsp, lgmdP[i].max_Tsp, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["Tsr"] = gaussPerturbation(lgmdP[i].ParamsDict["Tsr"], lgmdP[i].min_Tsr, lgmdP[i].max_Tsr, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["tau_E"] = gaussPerturbation(lgmdP[i].ParamsDict["tau_E"], lgmdP[i].min_tau_cen_E, lgmdP[i].max_tau_cen_E, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["tau_hp"] = gaussPerturbation(lgmdP[i].ParamsDict["tau_hp"], lgmdP[i].min_tau_hp, lgmdP[i].max_tau_hp, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["W_base"] = gaussPerturbation(lgmdP[i].ParamsDict["W_base"], lgmdP[i].min_W_base, lgmdP[i].max_W_base, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["coe_sig"] = gaussPerturbation(lgmdP[i].ParamsDict["coe_sig"], lgmdP[i].min_Csig, lgmdP[i].max_Csig, sigma, gauss_scale);
                            lgmdP[i].ParamsDict["std_w"] = gaussPerturbation(lgmdP[i].ParamsDict["std_w"], lgmdP[i].min_std_w, lgmdP[i].max_std_w, sigma, gauss_scale);
                            lgmdP[i].LGMDPlus_searchingParametersExchanging();
                            objLGMDPlus.Add(lgmdP[i]);
                        }

                        for (int i = 1; i < initLGMDDAgentsNo; i++)
                        {
                            lgmdD[i] = new LGMDs(width, height, fps);
                            lgmdD[i].ParamsDict["Tffi"] = gaussPerturbation(lgmdD[i].ParamsDict["Tffi"], lgmdD[i].min_Tffi, lgmdD[i].max_Tffi, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["Ts"] = gaussPerturbation(lgmdD[i].ParamsDict["Ts"], lgmdD[i].min_Ts, lgmdD[i].max_Ts, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["Tsp"] = gaussPerturbation(lgmdD[i].ParamsDict["Tsp"], lgmdD[i].min_Tsp, lgmdD[i].max_Tsp, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["coe_sig"] = gaussPerturbation(lgmdD[i].ParamsDict["coe_sig"], lgmdD[i].min_Csig, lgmdD[i].max_Csig, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["W_i_off"] = gaussPerturbation(lgmdD[i].ParamsDict["W_i_off"], lgmdD[i].min_W_i_off, lgmdD[i].max_W_i_off, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["W_i_on"] = gaussPerturbation(lgmdD[i].ParamsDict["W_i_on"], lgmdD[i].min_W_i_on, lgmdD[i].max_W_i_on, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["tau_hp"] = gaussPerturbation(lgmdD[i].ParamsDict["tau_hp"], lgmdD[i].min_tau_hp, lgmdD[i].max_tau_hp, sigma, gauss_scale);
                            lgmdD[i].ParamsDict["tau_lp"] = gaussPerturbation(lgmdD[i].ParamsDict["tau_lp"], lgmdD[i].min_tau_lp, lgmdD[i].max_tau_lp, sigma, gauss_scale);
                            lgmdD[i].LGMDD_searchingParametersExchanging();
                            objLGMDD.Add(lgmdD[i]);
                        }

                        for (int i = 1; i < initLGMDSAgentsNo; i++)
                        {
                            lgmdS[i] = new LGMDSingle(width, height);
                            lgmdS[i].ParamsDict["Tffi"] = gaussPerturbation(lgmdS[i].ParamsDict["Tffi"], lgmdS[i].min_Tffi, lgmdS[i].max_Tffi, sigma, gauss_scale);
                            lgmdS[i].ParamsDict["Ts"] = gaussPerturbation(lgmdS[i].ParamsDict["Ts"], lgmdS[i].min_Ts, lgmdS[i].max_Ts, sigma, gauss_scale);
                            lgmdS[i].ParamsDict["Tsp"] = gaussPerturbation(lgmdS[i].ParamsDict["Tsp"], lgmdS[i].min_Tsp, lgmdS[i].max_Tsp, sigma, gauss_scale);
                            lgmdS[i].ParamsDict["Wi"] = gaussPerturbation(lgmdS[i].ParamsDict["Wi"], lgmdS[i].min_Wi, lgmdS[i].max_Wi, sigma, gauss_scale);
                            lgmdS[i].ParamsDict["coe_sig"] = gaussPerturbation(lgmdS[i].ParamsDict["coe_sig"], lgmdS[i].min_Csig, lgmdS[i].max_Csig, sigma, gauss_scale);
                            lgmdS[i].LGMDS_searchingParametersExchanging();
                            objLGMDS.Add(lgmdS[i]);
                        }
                        Console.WriteLine("GAs population initialisation: Mixed");
                        break;
                    }
                default:
                    break;
            }
        }

        /// <summary>
        /// Run training dataset
        /// </summary>
        /// <param name="img1"></param>
        /// <param name="img2"></param>
        /// <param name="t"></param>
        private void runDataset(byte[, ,] img1, byte[, ,] img2, int t)
        {
            if (objLGMDPlus.Count > 0)  // LGMDPlus models processing
            {
                foreach (LGMDPlus lgmdP in objLGMDPlus)
                {
                    if (!lgmdP.IsOldGeneration) // new generation
                        lgmdP.LGMDPlus_Processing(img1, img2, t);
                }
            }
            if (objLGMDS.Count > 0) // LGMD TNN models processing
            {
                foreach (LGMDSingle lgmdS in objLGMDS)
                {
                    if (!lgmdS.IsOldGeneration) //new generation
                        lgmdS.LGMDSProcessing(img1, img2, t);
                }
            }
            if (objLGMDD.Count > 0) // LGMD NN models processing
            {
                foreach (LGMDs lgmdD in objLGMDD)
                {
                    if (!lgmdD.IsOldGeneration) // new generation
                        lgmdD.LGMDsCircuitry(img1, img2, t);
                }
            }
        }

        /// <summary>
        /// Mutation function for new generation of agents
        /// </summary>
        private void Mutation()
        {
            double rand;
            switch (kindGA)
            {
                case 1: // mutation in LGMDPlus model new popupation only
                    {
                        // select new population for mutation with global possibility Pm at each gene
                        foreach (LGMDPlus lgmdP in objLGMDPlus)
                        {
                            if (!lgmdP.IsOldGeneration) // new generation
                            {
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // Tffi
                                    lgmdP.ParamsDict["Tffi"] = gaussPerturbation(lgmdP.ParamsDict["Tffi"], lgmdP.min_Tffi, lgmdP.max_Tffi, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // Tde
                                    lgmdP.ParamsDict["Tde"] = gaussPerturbation(lgmdP.ParamsDict["Tde"], lgmdP.min_Tde, lgmdP.max_Tde, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // Tsp
                                    lgmdP.ParamsDict["Tsp"] = gaussPerturbation(lgmdP.ParamsDict["Tsp"], lgmdP.min_Tsp, lgmdP.max_Tsp, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // Tsr
                                    lgmdP.ParamsDict["Tsr"] = gaussPerturbation(lgmdP.ParamsDict["Tsr"], lgmdP.min_Tsr, lgmdP.max_Tsr, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // tau_E
                                    lgmdP.ParamsDict["tau_E"] = gaussPerturbation(lgmdP.ParamsDict["tau_E"], lgmdP.min_tau_cen_E, lgmdP.max_tau_cen_E, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // tau_hp
                                    lgmdP.ParamsDict["tau_hp"] = gaussPerturbation(lgmdP.ParamsDict["tau_hp"], lgmdP.min_tau_hp, lgmdP.max_tau_hp, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // W_base
                                    lgmdP.ParamsDict["W_base"] = gaussPerturbation(lgmdP.ParamsDict["W_base"], lgmdP.min_W_base, lgmdP.max_W_base, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // coe_sig
                                    lgmdP.ParamsDict["coe_sig"] = gaussPerturbation(lgmdP.ParamsDict["coe_sig"], lgmdP.min_Csig, lgmdP.max_Csig, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // std_w
                                    lgmdP.ParamsDict["std_w"] = gaussPerturbation(lgmdP.ParamsDict["std_w"], lgmdP.min_std_w, lgmdP.max_std_w, sigma, gauss_scale);
                                // exchanging parameters to model space after mutation
                                lgmdP.LGMDPlus_searchingParametersExchanging();
                            }
                            else
                                continue;
                        }
                        Console.WriteLine("Mutation: LGMDPlus new population");
                        break;
                    }
                case 2: // mutation in LGMD TNN model new population only
                    {
                        // select new population for mutation with global possibility Pm at each gene
                        foreach (LGMDSingle lgmdS in objLGMDS)
                        {
                            if (!lgmdS.IsOldGeneration) // new generation
                            {
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // Tffi
                                    lgmdS.ParamsDict["Tffi"] = gaussPerturbation(lgmdS.ParamsDict["Tffi"], lgmdS.min_Tffi, lgmdS.max_Tffi, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // Ts
                                    lgmdS.ParamsDict["Ts"] = gaussPerturbation(lgmdS.ParamsDict["Ts"], lgmdS.min_Ts, lgmdS.max_Ts, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // Tsp
                                    lgmdS.ParamsDict["Tsp"] = gaussPerturbation(lgmdS.ParamsDict["Tsp"], lgmdS.min_Tsp, lgmdS.max_Tsp, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // Wi
                                    lgmdS.ParamsDict["Wi"] = gaussPerturbation(lgmdS.ParamsDict["Wi"], lgmdS.min_Wi, lgmdS.max_Wi, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // coe_sig
                                    lgmdS.ParamsDict["coe_sig"] = gaussPerturbation(lgmdS.ParamsDict["coe_sig"], lgmdS.min_Csig, lgmdS.max_Csig, sigma, gauss_scale);
                                // exchanging parameters to model space after mutation
                                lgmdS.LGMDS_searchingParametersExchanging();
                            }
                            else
                                continue;
                        }
                        Console.WriteLine("Mutation: LGMD TNN new population");
                        break;
                    }
                case 3: // mutation in LGMD NN model new population only
                    {
                        // select new population for mutation with global possibility Pm at each gene
                        foreach (LGMDs lgmdD in objLGMDD)
                        {
                            if (!lgmdD.IsOldGeneration) // new generation
                            {
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // Tffi
                                    lgmdD.ParamsDict["Tffi"] = gaussPerturbation(lgmdD.ParamsDict["Tffi"], lgmdD.min_Tffi, lgmdD.max_Tffi, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // Ts
                                    lgmdD.ParamsDict["Ts"] = gaussPerturbation(lgmdD.ParamsDict["Ts"], lgmdD.min_Ts, lgmdD.max_Ts, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // Tsp
                                    lgmdD.ParamsDict["Tsp"] = gaussPerturbation(lgmdD.ParamsDict["Tsp"], lgmdD.min_Tsp, lgmdD.max_Tsp, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // coe_sig
                                    lgmdD.ParamsDict["coe_sig"] = gaussPerturbation(lgmdD.ParamsDict["coe_sig"], lgmdD.min_Csig, lgmdD.max_Csig, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // W_i_off
                                    lgmdD.ParamsDict["W_i_off"] = gaussPerturbation(lgmdD.ParamsDict["W_i_off"], lgmdD.min_W_i_off, lgmdD.max_W_i_off, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // W_i_on
                                    lgmdD.ParamsDict["W_i_on"] = gaussPerturbation(lgmdD.ParamsDict["W_i_on"], lgmdD.min_W_i_on, lgmdD.max_W_i_on, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // tau_hp
                                    lgmdD.ParamsDict["tau_hp"] = gaussPerturbation(lgmdD.ParamsDict["tau_hp"], lgmdD.min_tau_hp, lgmdD.max_tau_hp, sigma, gauss_scale);
                                rand = _rand.NextDouble();
                                if (rand <= Pm)    // tau_lp
                                    lgmdD.ParamsDict["tau_lp"] = gaussPerturbation(lgmdD.ParamsDict["tau_lp"], lgmdD.min_tau_lp, lgmdD.max_tau_lp, sigma, gauss_scale);
                                // exchanging parameters to model space after mutation
                                lgmdD.LGMDD_searchingParametersExchanging();
                            }
                            else
                                continue;
                        }
                        Console.WriteLine("Mutation: LGMD NN new population");
                        break;
                    }
                case 4: // mutation in mixed population
                    {
                        if (objLGMDPlus.Count > 0)  // LGMDPlus population
                        {
                            // select new population for mutation with global possibility Pm at each gene
                            foreach (LGMDPlus lgmdP in objLGMDPlus)
                            {
                                if (!lgmdP.IsOldGeneration) // new generation
                                {
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // Tffi
                                        lgmdP.ParamsDict["Tffi"] = gaussPerturbation(lgmdP.ParamsDict["Tffi"], lgmdP.min_Tffi, lgmdP.max_Tffi, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // Tde
                                        lgmdP.ParamsDict["Tde"] = gaussPerturbation(lgmdP.ParamsDict["Tde"], lgmdP.min_Tde, lgmdP.max_Tde, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // Tsp
                                        lgmdP.ParamsDict["Tsp"] = gaussPerturbation(lgmdP.ParamsDict["Tsp"], lgmdP.min_Tsp, lgmdP.max_Tsp, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // Tsr
                                        lgmdP.ParamsDict["Tsr"] = gaussPerturbation(lgmdP.ParamsDict["Tsr"], lgmdP.min_Tsr, lgmdP.max_Tsr, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // tau_E
                                        lgmdP.ParamsDict["tau_E"] = gaussPerturbation(lgmdP.ParamsDict["tau_E"], lgmdP.min_tau_cen_E, lgmdP.max_tau_cen_E, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // tau_hp
                                        lgmdP.ParamsDict["tau_hp"] = gaussPerturbation(lgmdP.ParamsDict["tau_hp"], lgmdP.min_tau_hp, lgmdP.max_tau_hp, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // W_base
                                        lgmdP.ParamsDict["W_base"] = gaussPerturbation(lgmdP.ParamsDict["W_base"], lgmdP.min_W_base, lgmdP.max_W_base, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // coe_sig
                                        lgmdP.ParamsDict["coe_sig"] = gaussPerturbation(lgmdP.ParamsDict["coe_sig"], lgmdP.min_Csig, lgmdP.max_Csig, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // std_w
                                        lgmdP.ParamsDict["std_w"] = gaussPerturbation(lgmdP.ParamsDict["std_w"], lgmdP.min_std_w, lgmdP.max_std_w, sigma, gauss_scale);
                                    // exchanging parameters to model space after mutation
                                    lgmdP.LGMDPlus_searchingParametersExchanging();
                                }
                                else
                                    continue;
                            }
                        }
                        if (objLGMDS.Count > 0)   // LGMD TNN population
                        {
                            // select new population for mutation with global possibility Pm at each gene
                            foreach (LGMDSingle lgmdS in objLGMDS)
                            {
                                if (!lgmdS.IsOldGeneration) // new generation
                                {
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // Tffi
                                        lgmdS.ParamsDict["Tffi"] = gaussPerturbation(lgmdS.ParamsDict["Tffi"], lgmdS.min_Tffi, lgmdS.max_Tffi, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // Ts
                                        lgmdS.ParamsDict["Ts"] = gaussPerturbation(lgmdS.ParamsDict["Ts"], lgmdS.min_Ts, lgmdS.max_Ts, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // Tsp
                                        lgmdS.ParamsDict["Tsp"] = gaussPerturbation(lgmdS.ParamsDict["Tsp"], lgmdS.min_Tsp, lgmdS.max_Tsp, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // Wi
                                        lgmdS.ParamsDict["Wi"] = gaussPerturbation(lgmdS.ParamsDict["Wi"], lgmdS.min_Wi, lgmdS.max_Wi, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // coe_sig
                                        lgmdS.ParamsDict["coe_sig"] = gaussPerturbation(lgmdS.ParamsDict["coe_sig"], lgmdS.min_Csig, lgmdS.max_Csig, sigma, gauss_scale);
                                    // exchanging parameters to model space after mutation
                                    lgmdS.LGMDS_searchingParametersExchanging();
                                }
                                else
                                    continue;
                            }
                        }
                        if (objLGMDD.Count > 0)   // LGMD NN population
                        {
                            // select new population for mutation with global possibility Pm at each gene
                            foreach (LGMDs lgmdD in objLGMDD)
                            {
                                if (!lgmdD.IsOldGeneration) // new generation
                                {
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // Tffi
                                        lgmdD.ParamsDict["Tffi"] = gaussPerturbation(lgmdD.ParamsDict["Tffi"], lgmdD.min_Tffi, lgmdD.max_Tffi, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // Ts
                                        lgmdD.ParamsDict["Ts"] = gaussPerturbation(lgmdD.ParamsDict["Ts"], lgmdD.min_Ts, lgmdD.max_Ts, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // Tsp
                                        lgmdD.ParamsDict["Tsp"] = gaussPerturbation(lgmdD.ParamsDict["Tsp"], lgmdD.min_Tsp, lgmdD.max_Tsp, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // coe_sig
                                        lgmdD.ParamsDict["coe_sig"] = gaussPerturbation(lgmdD.ParamsDict["coe_sig"], lgmdD.min_Csig, lgmdD.max_Csig, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // W_i_off
                                        lgmdD.ParamsDict["W_i_off"] = gaussPerturbation(lgmdD.ParamsDict["W_i_off"], lgmdD.min_W_i_off, lgmdD.max_W_i_off, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // W_i_on
                                        lgmdD.ParamsDict["W_i_on"] = gaussPerturbation(lgmdD.ParamsDict["W_i_on"], lgmdD.min_W_i_on, lgmdD.max_W_i_on, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // tau_hp
                                        lgmdD.ParamsDict["tau_hp"] = gaussPerturbation(lgmdD.ParamsDict["tau_hp"], lgmdD.min_tau_hp, lgmdD.max_tau_hp, sigma, gauss_scale);
                                    rand = _rand.NextDouble();
                                    if (rand <= Pm)    // tau_lp
                                        lgmdD.ParamsDict["tau_lp"] = gaussPerturbation(lgmdD.ParamsDict["tau_lp"], lgmdD.min_tau_lp, lgmdD.max_tau_lp, sigma, gauss_scale);
                                    // exchanging parameters to model space after mutation
                                    lgmdD.LGMDD_searchingParametersExchanging();
                                }
                                else
                                    continue;
                            }
                        }
                        Console.WriteLine("Mutation: mixed population");
                        break;
                    }
                default:
                    break;
            }
        }

        /// <summary>
        /// Crossover function for selected pairwise parent agents
        /// </summary>
        private void Crossover()
        {
            switch (kindGA)
            {
                case 1: // only LGMDPlus model population evolves
                    {
                        LGMDPlus[] new_lgmdP = new LGMDPlus[Npa / 2];
                        List<LGMDPlus> lgmdP_parents = new List<LGMDPlus>();
                        foreach (LGMDPlus lgmdP in objLGMDPlus)
                        {
                            if (lgmdP.IsParent) // is parent agent
                            {
                                lgmdP_parents.Add(lgmdP);
                                lgmdP.IsParent = false;
                            }
                        }
                        // for debug use
                        //Console.WriteLine(lgmdP_parents.Count);
                        // sort by fitness
                        lgmdP_parents.Sort();
                        for (int i = 0; i < Npa / 2; i++)
                        {
                            new_lgmdP[i] = new LGMDPlus(width, height, fps);
                            float bias = lgmdP_parents[i].Fitness / (lgmdP_parents[i].Fitness + lgmdP_parents[i + Npa / 2].Fitness);
                            double rand1 = _rand.NextDouble();
                            if (rand1 > this.Pc) // no crossover
                            {
                                // chromosome equals to better fitting agent
                                foreach (KeyValuePair<string, float> pair in lgmdP_parents[i].ParamsDict)
                                {
                                    new_lgmdP[i].ParamsDict[pair.Key] = lgmdP_parents[i].ParamsDict[pair.Key];
                                }
                            }
                            else // uniform crossover method
                            {
                                foreach (KeyValuePair<string, float> pair in lgmdP_parents[i].ParamsDict)
                                {
                                    double rand2 = _rand.NextDouble();
                                    if (rand2 > bias)
                                        new_lgmdP[i].ParamsDict[pair.Key] = lgmdP_parents[i + Npa / 2].ParamsDict[pair.Key];
                                    else
                                        new_lgmdP[i].ParamsDict[pair.Key] = lgmdP_parents[i].ParamsDict[pair.Key];
                                }
                            }
                            // attention
                            objLGMDPlus.Add(new_lgmdP[i]);
                        }
                        Console.WriteLine("Crossover: LGMDPlus parent agents");
                        break;
                    }
                case 2: // only LGMD TNN model population evolves
                    {
                        LGMDSingle[] new_lgmdS = new LGMDSingle[Npa / 2];
                        List<LGMDSingle> lgmdS_parents = new List<LGMDSingle>();
                        foreach (LGMDSingle lgmdS in objLGMDS)
                        {
                            if (lgmdS.IsParent) // is parent agent
                            {
                                lgmdS_parents.Add(lgmdS);
                                lgmdS.IsParent = false;
                            }
                        }
                        // sort by fitness
                        lgmdS_parents.Sort();
                        for (int i = 0; i < Npa / 2; i++)
                        {
                            new_lgmdS[i] = new LGMDSingle(width, height);
                            float bias = lgmdS_parents[i].Fitness / (lgmdS_parents[i].Fitness + lgmdS_parents[i + Npa / 2].Fitness);
                            double rand1 = _rand.NextDouble();
                            if (rand1 > this.Pc) // no crossover
                            {
                                // chromosome equals to better fitting agent
                                foreach (KeyValuePair<string, float> pair in lgmdS_parents[i].ParamsDict)
                                {
                                    new_lgmdS[i].ParamsDict[pair.Key] = lgmdS_parents[i].ParamsDict[pair.Key];
                                }
                            }
                            else // uniform crossover method
                            {
                                foreach (KeyValuePair<string, float> pair in lgmdS_parents[i].ParamsDict)
                                {
                                    double rand2 = _rand.NextDouble();
                                    if (rand2 > bias)
                                        new_lgmdS[i].ParamsDict[pair.Key] = lgmdS_parents[i + Npa / 2].ParamsDict[pair.Key];
                                    else
                                        new_lgmdS[i].ParamsDict[pair.Key] = lgmdS_parents[i].ParamsDict[pair.Key];
                                }
                            }
                            // attention
                            objLGMDS.Add(new_lgmdS[i]);
                        }
                        Console.WriteLine("Crossover: LGMD TNN parent agents");
                        break;
                    }
                case 3: // only LGMD NN model population evolves
                    {
                        LGMDs[] new_lgmdD = new LGMDs[Npa / 2];
                        List<LGMDs> lgmdD_parents = new List<LGMDs>();
                        foreach (LGMDs lgmdD in objLGMDD)
                        {
                            if (lgmdD.IsParent) // is parent agent
                            {
                                lgmdD_parents.Add(lgmdD);
                                lgmdD.IsParent = false;
                            }
                        }
                        // sort by fitness
                        lgmdD_parents.Sort();
                        for (int i = 0; i < Npa / 2; i++)
                        {
                            new_lgmdD[i] = new LGMDs(width, height, fps);
                            float bias = lgmdD_parents[i].Fitness / (lgmdD_parents[i].Fitness + lgmdD_parents[i + Npa / 2].Fitness);
                            double rand1 = _rand.NextDouble();
                            if (rand1 > this.Pc) // no crossover
                            {
                                // chromosome equals to better fitting agent
                                foreach (KeyValuePair<string, float> pair in lgmdD_parents[i].ParamsDict)
                                {
                                    new_lgmdD[i].ParamsDict[pair.Key] = lgmdD_parents[i].ParamsDict[pair.Key];
                                }
                            }
                            else // uniform crossover method
                            {
                                foreach (KeyValuePair<string, float> pair in lgmdD_parents[i].ParamsDict)
                                {
                                    double rand2 = _rand.NextDouble();
                                    if (rand2 > bias)
                                        new_lgmdD[i].ParamsDict[pair.Key] = lgmdD_parents[i + Npa / 2].ParamsDict[pair.Key];
                                    else
                                        new_lgmdD[i].ParamsDict[pair.Key] = lgmdD_parents[i].ParamsDict[pair.Key];
                                }
                            }
                            // attention
                            objLGMDD.Add(new_lgmdD[i]);
                        }
                        Console.WriteLine("Crossover: LGMD NN parent agents");
                        break;
                    }
                case 4: // coevolution
                    {
                        // init number of parents for each population
                        int Npa_lgmdplus = objLGMDPlus.Count - objLGMDPlus.Count % 2;
                        if (Npa_lgmdplus > Npa)
                            Npa_lgmdplus = Npa;
                        int Npa_lgmdtnn = objLGMDS.Count - objLGMDS.Count % 2;
                        if (Npa_lgmdtnn > Npa)
                            Npa_lgmdtnn = Npa;
                        int Npa_lgmdnn = objLGMDD.Count - objLGMDD.Count % 2;
                        if (Npa_lgmdnn > Npa)
                            Npa_lgmdnn = Npa;
                        // init objects
                        List<LGMDPlus> lgmdP_parents = new List<LGMDPlus>();
                        List<LGMDSingle> lgmdS_parents = new List<LGMDSingle>();
                        List<LGMDs> lgmdD_parents = new List<LGMDs>();
                        // select LGMDPlus parent agents
                        if (Npa_lgmdplus > 1)
                        {
                            foreach (LGMDPlus lgmdP in objLGMDPlus)
                            {
                                if (lgmdP.IsParent) // is parent agent
                                {
                                    lgmdP_parents.Add(lgmdP);
                                    lgmdP.IsParent = false;
                                }
                            }
                        }
                        // check parent agents in LGMDPlus model population
                        if (lgmdP_parents.Count > 0)
                        {
                            LGMDPlus[] new_lgmdP = new LGMDPlus[Npa_lgmdplus / 2];
                            // sort by fitness
                            lgmdP_parents.Sort();
                            for (int i = 0; i < Npa_lgmdplus / 2; i++)
                            {
                                new_lgmdP[i] = new LGMDPlus(width, height, fps);
                                float bias = lgmdP_parents[i].Fitness / (lgmdP_parents[i].Fitness + lgmdP_parents[i + Npa_lgmdplus / 2].Fitness);
                                double rand1 = _rand.NextDouble();
                                if (rand1 > this.Pc) // no crossover
                                {
                                    // chromosome equals to better fitting agent
                                    foreach (KeyValuePair<string, float> pair in lgmdP_parents[i].ParamsDict)
                                    {
                                        new_lgmdP[i].ParamsDict[pair.Key] = lgmdP_parents[i].ParamsDict[pair.Key];
                                    }
                                }
                                else // uniform crossover method
                                {
                                    foreach (KeyValuePair<string, float> pair in lgmdP_parents[i].ParamsDict)
                                    {
                                        double rand2 = _rand.NextDouble();
                                        if (rand2 > bias)
                                            new_lgmdP[i].ParamsDict[pair.Key] = lgmdP_parents[i + Npa_lgmdplus / 2].ParamsDict[pair.Key];
                                        else
                                            new_lgmdP[i].ParamsDict[pair.Key] = lgmdP_parents[i].ParamsDict[pair.Key];
                                    }
                                }
                                objLGMDPlus.Add(new_lgmdP[i]);
                            }
                        }
                        // select LGMD TNN parent agents
                        if (Npa_lgmdtnn > 1)
                        {
                            foreach (LGMDSingle lgmdS in objLGMDS)
                            {
                                if (lgmdS.IsParent) // is parent agent
                                {
                                    lgmdS_parents.Add(lgmdS);
                                    lgmdS.IsParent = false;
                                }
                            }
                        }
                        // check parent agents in LGMD TNN model population
                        if (lgmdS_parents.Count > 0)
                        {
                            LGMDSingle[] new_lgmdS = new LGMDSingle[Npa_lgmdtnn / 2];
                            lgmdS_parents.Sort();
                            for (int i = 0; i < Npa_lgmdtnn / 2; i++)
                            {
                                new_lgmdS[i] = new LGMDSingle(width, height);
                                float bias = lgmdS_parents[i].Fitness / (lgmdS_parents[i].Fitness + lgmdS_parents[i + Npa_lgmdtnn / 2].Fitness);
                                double rand1 = _rand.NextDouble();
                                if (rand1 > this.Pc) // no crossover
                                {
                                    // chromosome equals to better fitting agent
                                    foreach (KeyValuePair<string, float> pair in lgmdS_parents[i].ParamsDict)
                                    {
                                        new_lgmdS[i].ParamsDict[pair.Key] = lgmdS_parents[i].ParamsDict[pair.Key];
                                    }
                                }
                                else // uniform crossover method
                                {
                                    foreach (KeyValuePair<string, float> pair in lgmdS_parents[i].ParamsDict)
                                    {
                                        double rand2 = _rand.NextDouble();
                                        if (rand2 > bias)
                                            new_lgmdS[i].ParamsDict[pair.Key] = lgmdS_parents[i + Npa_lgmdtnn / 2].ParamsDict[pair.Key];
                                        else
                                            new_lgmdS[i].ParamsDict[pair.Key] = lgmdS_parents[i].ParamsDict[pair.Key];
                                    }
                                }
                                objLGMDS.Add(new_lgmdS[i]);
                            }
                        }
                        // select LGMD NN parent agents
                        if (Npa_lgmdnn > 1)
                        {
                            if (objLGMDD.Count >= Npa)
                            {
                                foreach (LGMDs lgmdD in objLGMDD)
                                {
                                    if (lgmdD.IsParent) // is parent agent
                                    {
                                        lgmdD_parents.Add(lgmdD);
                                        lgmdD.IsParent = false;
                                    }
                                }
                            }
                        }
                        // check parent agents in LGMD NN model population
                        if (lgmdD_parents.Count > 0)
                        {
                            LGMDs[] new_lgmdD = new LGMDs[Npa_lgmdnn / 2];
                            lgmdD_parents.Sort();
                            for (int i = 0; i < Npa_lgmdnn / 2; i++)
                            {
                                new_lgmdD[i] = new LGMDs(width, height, fps);
                                float bias = lgmdD_parents[i].Fitness / (lgmdD_parents[i].Fitness + lgmdD_parents[i + Npa_lgmdnn / 2].Fitness);
                                double rand1 = _rand.NextDouble();
                                if (rand1 > this.Pc) // no crossover
                                {
                                    // chromosome equals to better fitting agent
                                    foreach (KeyValuePair<string, float> pair in lgmdD_parents[i].ParamsDict)
                                    {
                                        new_lgmdD[i].ParamsDict[pair.Key] = lgmdD_parents[i].ParamsDict[pair.Key];
                                    }
                                }
                                else // uniform crossover method
                                {
                                    foreach (KeyValuePair<string, float> pair in lgmdD_parents[i].ParamsDict)
                                    {
                                        double rand2 = _rand.NextDouble();
                                        if (rand2 > bias)
                                            new_lgmdD[i].ParamsDict[pair.Key] = lgmdD_parents[i + Npa_lgmdnn / 2].ParamsDict[pair.Key];
                                        else
                                            new_lgmdD[i].ParamsDict[pair.Key] = lgmdD_parents[i].ParamsDict[pair.Key];
                                    }
                                }
                                objLGMDD.Add(new_lgmdD[i]);
                            }
                        }

                        Console.WriteLine("Crossover: mixed parent agents");
                        break;
                    }
                default:
                    break;
            }
        }

        /// <summary>
        /// Calculation of fitness value
        /// </summary>
        /// <param name="Fcol"></param>
        /// <param name="Fnon"></param>
        /// <param name="colFScore"></param>
        /// <param name="nonFScore"></param>
        /// <returns></returns>
        private float fitnessCalc(int Fcol, int Fnon, float colFScore, float nonFScore)
        {
            return (1 - (Fcol * colFScore + Fnon * nonFScore) / fitnessFScore) * 100;   // 100%
        }

        /// <summary>
        /// Selection of best agents as parents
        /// </summary>
        private void parentSelection()
        {
            switch (kindGA)
            {
                case 1: // LGMDPlus model population
                    {
                        // sort population by fitness value in a descending order
                        objLGMDPlus.Sort();
                        // for debug use
                        /*
                        foreach (LGMDPlus lgmdP in objLGMDPlus)
                        {
                            Console.WriteLine(lgmdP.Fitness);
                        }
                        */
                        // select top ranking agents as parents
                        for (int count = 0; count < Npa; count++)
                        {
                            objLGMDPlus[count].IsParent = true;
                        }
                        Console.WriteLine("Parent Selection: LGMDPlus agents population");
                        break;
                    }
                case 2: // LGMD TNN model polulation
                    {
                        // sort population by fitness value in a descending order
                        objLGMDS.Sort();
                        // select top ranking agents as parents
                        for (int count = 0; count < Npa; count++)
                        {
                            objLGMDS[count].IsParent = true;
                        }
                        Console.WriteLine("Parent Selection: LGMD TNN agents population");
                        break;
                    }
                case 3: // LGMD NN model population
                    {
                        // sort population by fitness value in a descending order
                        objLGMDD.Sort();
                        // select top ranking agents as parents
                        for (int count = 0; count < Npa; count++)
                        {
                            objLGMDD[count].IsParent = true;
                        }
                        Console.WriteLine("Parent Selection: LGMD NN agents population");
                        break;
                    }
                case 4: // mixed population
                    {
                        // init number of parents for each population
                        int Npa_lgmdplus = objLGMDPlus.Count - objLGMDPlus.Count % 2;
                        if (Npa_lgmdplus > Npa)
                            Npa_lgmdplus = Npa;
                        int Npa_lgmdtnn = objLGMDS.Count - objLGMDS.Count % 2;
                        if (Npa_lgmdtnn > Npa)
                            Npa_lgmdtnn = Npa;
                        int Npa_lgmdnn = objLGMDD.Count - objLGMDD.Count % 2;
                        if (Npa_lgmdnn > Npa)
                            Npa_lgmdnn = Npa;
                        // compare mean fitness value of three populations of agents, select parent agents from top ranking population
                        float minAvg = meanFitness.Min();
                        int index = 0;
                        for (int i = 0; i < meanFitness.Length; i++)
                        {
                            if (meanFitness[i] == minAvg)
                                index = i;
                        }
                        switch (index)
                        {
                            case 0: // compare LGMD TNN model and LGMD NN model populations
                                {
                                    if (meanFitness[1] > meanFitness[2]) // LGMD TNN model population wins
                                    {
                                        if (Npa_lgmdtnn > 0)    // LGMD TNN model population has agents
                                        {
                                            // sort population by fitness value in a descending order
                                            objLGMDS.Sort();
                                            // select top ranking agents as parents
                                            for (int count = 0; count < Npa_lgmdtnn; count++)
                                            {
                                                objLGMDS[count].IsParent = true;
                                            }
                                        }
                                        else  // LGMD TNN model has no agents, then check LGMD NN model population
                                        {
                                            if (Npa_lgmdnn > 0) //LGMD NN model population has agents
                                            {
                                                // sort population by fitness value in a descending order
                                                objLGMDD.Sort();
                                                // select top ranking agents as parents
                                                for (int count = 0; count < Npa_lgmdnn; count++)
                                                {
                                                    objLGMDD[count].IsParent = true;
                                                }
                                            }
                                            else  // neither of LGMD TNN/NN has agents, then select LGMD Plus model population
                                            {
                                                // sort population by fitness value in a descending order
                                                objLGMDPlus.Sort();
                                                // select top ranking agents as parents
                                                for (int count = 0; count < Npa_lgmdplus; count++)
                                                {
                                                    objLGMDPlus[count].IsParent = true;
                                                }
                                            }
                                        }
                                    }
                                    else  // LGMD NN model population wins
                                    {
                                        if (Npa_lgmdnn > 0) // LGMD NN model population has agents
                                        {
                                            // sort population by fitness value in a descending order
                                            objLGMDD.Sort();
                                            // select top ranking agents as parents
                                            for (int count = 0; count < Npa_lgmdnn; count++)
                                            {
                                                objLGMDD[count].IsParent = true;
                                            }
                                        }
                                        else    // LGMD NN population has no agents, then check LGMD TNN population
                                        {
                                            if (Npa_lgmdtnn > 0)    // LGMD TNN population has agents
                                            {
                                                // sort population by fitness value in a descending order
                                                objLGMDS.Sort();
                                                // select top ranking agents as parents
                                                for (int count = 0; count < Npa_lgmdtnn; count++)
                                                {
                                                    objLGMDS[count].IsParent = true;
                                                }
                                            }
                                            else    // neither has agents, then select LGMD Plus model population
                                            {
                                                // sort population by fitness value in a descending order
                                                objLGMDPlus.Sort();
                                                // select top ranking agents as parents
                                                for (int count = 0; count < Npa_lgmdplus; count++)
                                                {
                                                    objLGMDPlus[count].IsParent = true;
                                                }
                                            }
                                        }
                                    }
                                    break;
                                }
                            case 1: // compare LGMDPlus model and LGMD NN model populations
                                {
                                    if (meanFitness[0] > meanFitness[2]) // LGMD Plus model population wins
                                    {
                                        if (Npa_lgmdplus > 0)    // LGMD Plus model population has agents
                                        {
                                            // sort population by fitness value in a descending order
                                            objLGMDPlus.Sort();
                                            // select top ranking agents as parents
                                            for (int count = 0; count < Npa_lgmdplus; count++)
                                            {
                                                objLGMDPlus[count].IsParent = true;
                                            }
                                        }
                                        else  // LGMD Plus model has no agents, then check LGMD NN model population
                                        {
                                            if (Npa_lgmdnn > 0) //LGMD NN model population has agents
                                            {
                                                // sort population by fitness value in a descending order
                                                objLGMDD.Sort();
                                                // select top ranking agents as parents
                                                for (int count = 0; count < Npa_lgmdnn; count++)
                                                {
                                                    objLGMDD[count].IsParent = true;
                                                }
                                            }
                                            else  // neither of LGMD Plus/NN has agents, then select LGMD TNN model population
                                            {
                                                // sort population by fitness value in a descending order
                                                objLGMDS.Sort();
                                                // select top ranking agents as parents
                                                for (int count = 0; count < Npa_lgmdtnn; count++)
                                                {
                                                    objLGMDS[count].IsParent = true;
                                                }
                                            }
                                        }
                                    }
                                    else  // LGMD NN model population wins
                                    {
                                        if (Npa_lgmdnn > 0) // LGMD NN model population has agents
                                        {
                                            // sort population by fitness value in a descending order
                                            objLGMDD.Sort();
                                            // select top ranking agents as parents
                                            for (int count = 0; count < Npa_lgmdnn; count++)
                                            {
                                                objLGMDD[count].IsParent = true;
                                            }
                                        }
                                        else    // LGMD NN population has no agents, then check LGMD Plus population
                                        {
                                            if (Npa_lgmdplus > 0)    // LGMD Plus population has agents
                                            {
                                                // sort population by fitness value in a descending order
                                                objLGMDPlus.Sort();
                                                // select top ranking agents as parents
                                                for (int count = 0; count < Npa_lgmdplus; count++)
                                                {
                                                    objLGMDPlus[count].IsParent = true;
                                                }
                                            }
                                            else    // neither has agents, then select LGMD TNN model population
                                            {
                                                // sort population by fitness value in a descending order
                                                objLGMDS.Sort();
                                                // select top ranking agents as parents
                                                for (int count = 0; count < Npa_lgmdtnn; count++)
                                                {
                                                    objLGMDS[count].IsParent = true;
                                                }
                                            }
                                        }
                                    }
                                    break;
                                }
                            case 2: // compare LGMDPlus model and LGMD TNN model populations
                                {
                                    if (meanFitness[0] > meanFitness[1]) // LGMD Plus model population wins
                                    {
                                        if (Npa_lgmdplus > 0)    // LGMD Plus model population has agents
                                        {
                                            // sort population by fitness value in a descending order
                                            objLGMDPlus.Sort();
                                            // select top ranking agents as parents
                                            for (int count = 0; count < Npa_lgmdplus; count++)
                                            {
                                                objLGMDPlus[count].IsParent = true;
                                            }
                                        }
                                        else  // LGMD Plus model has no agents, then check LGMD TNN model population
                                        {
                                            if (Npa_lgmdtnn > 0) //LGMD TNN model population has agents
                                            {
                                                // sort population by fitness value in a descending order
                                                objLGMDS.Sort();
                                                // select top ranking agents as parents
                                                for (int count = 0; count < Npa_lgmdtnn; count++)
                                                {
                                                    objLGMDS[count].IsParent = true;
                                                }
                                            }
                                            else  // neither of LGMD Plus/TNN has agents, then select LGMD NN model population
                                            {
                                                // sort population by fitness value in a descending order
                                                objLGMDD.Sort();
                                                // select top ranking agents as parents
                                                for (int count = 0; count < Npa_lgmdnn; count++)
                                                {
                                                    objLGMDD[count].IsParent = true;
                                                }
                                            }
                                        }
                                    }
                                    else  // LGMD TNN model population wins
                                    {
                                        if (Npa_lgmdtnn > 0) // LGMD TNN model population has agents
                                        {
                                            // sort population by fitness value in a descending order
                                            objLGMDS.Sort();
                                            // select top ranking agents as parents
                                            for (int count = 0; count < Npa_lgmdtnn; count++)
                                            {
                                                objLGMDS[count].IsParent = true;
                                            }
                                        }
                                        else    // LGMD TNN population has no agents, then check LGMD Plus population
                                        {
                                            if (Npa_lgmdplus > 0)    // LGMD Plus population has agents
                                            {
                                                // sort population by fitness value in a descending order
                                                objLGMDPlus.Sort();
                                                // select top ranking agents as parents
                                                for (int count = 0; count < Npa_lgmdplus; count++)
                                                {
                                                    objLGMDPlus[count].IsParent = true;
                                                }
                                            }
                                            else    // neither has agents, then select LGMD NN model population
                                            {
                                                // sort population by fitness value in a descending order
                                                objLGMDD.Sort();
                                                // select top ranking agents as parents
                                                for (int count = 0; count < Npa_lgmdnn; count++)
                                                {
                                                    objLGMDD[count].IsParent = true;
                                                }
                                            }
                                        }
                                    }
                                    break;
                                }
                        }
                        Console.WriteLine("Parent Selection: mixed population");
                        break;
                    }
                default:
                    break;
            }
        }

        /// <summary>
        /// Selection of survivors in each generation evolution
        /// </summary>
        private void survivorSelection()
        {
            // define number of worst agents equalling to the amount of new agents to vanish, remaining agents as survivors
            int Nagents = Npa / 2;
            switch (kindGA)
            {
                case 1: // LGMDPlus agents only
                    {
                        // sort population by fitness
                        objLGMDPlus.Sort();
                        // remove Npa/2 number of last ranked agents
                        objLGMDPlus.RemoveRange(objLGMDPlus.Count - Nagents, Nagents);
                        Console.WriteLine("Survivor selection: LGMDPlus population");
                        break;
                    }
                case 2: // LGMD TNN agents only
                    {
                        // sort population by fitness
                        objLGMDS.Sort();
                        // remove Npa/2 number of last ranked agents
                        objLGMDS.RemoveRange(objLGMDS.Count - Nagents, Nagents);
                        Console.WriteLine("Survivor selection: LGMD TNN population");
                        break;
                    }
                case 3: // LGMD NN agents only
                    {
                        // sort population by fitness
                        objLGMDD.Sort();
                        // remove Npa/2 number of last ranked agents
                        objLGMDD.RemoveRange(objLGMDD.Count - Nagents, Nagents);
                        Console.WriteLine("Survivor selection: LGMD NN population");
                        break;
                    }
                case 4: // mixed agents of three kinds of models
                    {
                        // init number of parents for each population
                        int Npa_lgmdplus = objLGMDPlus.Count - objLGMDPlus.Count % 2;
                        if (Npa_lgmdplus > Npa)
                            Npa_lgmdplus = Npa;
                        int Npa_lgmdtnn = objLGMDS.Count - objLGMDS.Count % 2;
                        if (Npa_lgmdtnn > Npa)
                            Npa_lgmdtnn = Npa;
                        int Npa_lgmdnn = objLGMDD.Count - objLGMDD.Count % 2;
                        if (Npa_lgmdnn > Npa)
                            Npa_lgmdnn = Npa;
                        int Npa_lgmdplus_des = Npa_lgmdplus / 2;
                        int Npa_lgmdtnn_des = Npa_lgmdtnn / 2;
                        int Npa_lgmdnn_des = Npa_lgmdnn / 2;
                        // compare mean fitness value of three populations of agents, remove the last ranking agents from the worst population
                        float minAvg = meanFitness.Min();
                        int index = 0;
                        for (int i = 0; i < meanFitness.Length; i++)
                        {
                            if (meanFitness[i] == minAvg)
                                index = i;
                        }
                        switch (index)
                        {
                            case 0: // LGMDPlus population performs worst
                                {
                                    if (objLGMDPlus.Count > 0)  // LGMDPlus agents existing
                                    {
                                        objLGMDPlus.Sort();
                                        if (objLGMDPlus.Count == 1)
                                            objLGMDPlus.RemoveAt(0);
                                        else
                                            objLGMDPlus.RemoveRange(objLGMDPlus.Count - Npa_lgmdplus_des, Npa_lgmdplus_des);
                                    }
                                    else // LGMDPlus agents vanished (meanFitness[0] = 0), compare other two populations
                                    {
                                        if (meanFitness[1] > meanFitness[2])    // LGMD TNN population performs better
                                        {
                                            if (objLGMDD.Count > 0) // LGMD NN agents existing
                                            {
                                                objLGMDD.Sort();
                                                if (objLGMDD.Count == 1)
                                                    objLGMDD.RemoveAt(0);
                                                else
                                                    objLGMDD.RemoveRange(objLGMDD.Count - Npa_lgmdnn_des, Npa_lgmdnn_des);
                                            }
                                            else // only left LGMD TNN agents
                                            {
                                                objLGMDS.Sort();
                                                if (objLGMDS.Count > totalAgents)
                                                    objLGMDS.RemoveRange(totalAgents, objLGMDS.Count - totalAgents);
                                                else
                                                    objLGMDS.RemoveRange(objLGMDS.Count - Npa_lgmdplus_des, Npa_lgmdplus_des);
                                            }
                                        }
                                        else // LGMD NN population performs better
                                        {
                                            if (objLGMDS.Count > 0) // LGMD TNN agents existing
                                            {
                                                objLGMDS.Sort();
                                                if (objLGMDS.Count == 1)
                                                    objLGMDS.RemoveAt(0);
                                                else
                                                    objLGMDS.RemoveRange(objLGMDS.Count - Npa_lgmdtnn_des, Npa_lgmdtnn_des);
                                            }
                                            else // only left LGMD NN agents
                                            {
                                                objLGMDD.Sort();
                                                if (objLGMDD.Count > totalAgents)
                                                    objLGMDD.RemoveRange(totalAgents, objLGMDD.Count - totalAgents);
                                                else
                                                    objLGMDD.RemoveRange(objLGMDD.Count - Npa_lgmdnn_des, Npa_lgmdnn_des);
                                            }
                                        }
                                    }
                                    break;
                                }
                            case 1: // LGMD TNN population performs worst
                                {
                                    if (objLGMDS.Count > 0)  // LGMD TNN agents existing
                                    {
                                        objLGMDS.Sort();
                                        if (objLGMDS.Count == 1)
                                            objLGMDS.RemoveAt(0);
                                        else
                                            objLGMDS.RemoveRange(objLGMDS.Count - Npa_lgmdtnn_des, Npa_lgmdtnn_des);
                                    }
                                    else // LGMD TNN agents vanished (meanFitness[1] = 0), compare other two populations
                                    {
                                        if (meanFitness[0] > meanFitness[2])    // LGMDPlus population performs better
                                        {
                                            if (objLGMDD.Count > 0) // LGMD NN agents existing
                                            {
                                                objLGMDD.Sort();
                                                if (objLGMDD.Count == 1)
                                                    objLGMDD.RemoveAt(0);
                                                else
                                                    objLGMDD.RemoveRange(objLGMDD.Count - Npa_lgmdnn_des, Npa_lgmdnn_des);
                                            }
                                            else // only left LGMDPlus agents
                                            {
                                                objLGMDPlus.Sort();
                                                if (objLGMDPlus.Count > totalAgents)
                                                    objLGMDPlus.RemoveRange(totalAgents, objLGMDPlus.Count - totalAgents);
                                                else
                                                    objLGMDPlus.RemoveRange(objLGMDPlus.Count - Npa_lgmdplus_des, Npa_lgmdplus_des);
                                            }
                                        }
                                        else // LGMD NN population performs better
                                        {
                                            if (objLGMDPlus.Count > 0) // LGMDPlus agents existing
                                            {
                                                objLGMDPlus.Sort();
                                                if (objLGMDPlus.Count == 1)
                                                    objLGMDPlus.RemoveAt(0);
                                                else
                                                    objLGMDPlus.RemoveRange(objLGMDPlus.Count - Npa_lgmdplus_des, Npa_lgmdplus_des);
                                            }
                                            else // only left LGMD NN agents
                                            {
                                                objLGMDD.Sort();
                                                if (objLGMDD.Count > totalAgents)
                                                    objLGMDD.RemoveRange(totalAgents, objLGMDD.Count - totalAgents);
                                                else
                                                    objLGMDD.RemoveRange(objLGMDD.Count - Npa_lgmdnn_des, Npa_lgmdnn_des);
                                            }
                                        }
                                    }
                                    break;
                                }
                            case 2: // LGMD NN population performs worst
                                {
                                    if (objLGMDD.Count > 0)  // LGMD NN agents existing
                                    {
                                        objLGMDD.Sort();
                                        if (objLGMDD.Count == 1)
                                            objLGMDD.RemoveAt(0);
                                        else
                                            objLGMDD.RemoveRange(objLGMDD.Count - Npa_lgmdnn_des, Npa_lgmdnn_des);
                                    }
                                    else // LGMD NN agents vanished (meanFitness[2] = 0), compare other two populations
                                    {
                                        if (meanFitness[0] > meanFitness[1])    // LGMDPlus population performs better
                                        {
                                            if (objLGMDS.Count > 0) // LGMD TNN agents existing
                                            {
                                                objLGMDS.Sort();
                                                if (objLGMDS.Count == 1)
                                                    objLGMDS.RemoveAt(0);
                                                else
                                                    objLGMDS.RemoveRange(objLGMDS.Count - Npa_lgmdtnn_des, Npa_lgmdtnn_des);
                                            }
                                            else // only left LGMDPlus agents
                                            {
                                                objLGMDPlus.Sort();
                                                if (objLGMDPlus.Count > totalAgents)
                                                    objLGMDPlus.RemoveRange(totalAgents, objLGMDPlus.Count - totalAgents);
                                                else
                                                    objLGMDPlus.RemoveRange(objLGMDPlus.Count - Npa_lgmdplus_des, Npa_lgmdplus_des);
                                            }
                                        }
                                        else // LGMD TNN population performs better
                                        {
                                            if (objLGMDPlus.Count > 0) // LGMDPlus agents existing 
                                            {
                                                objLGMDPlus.Sort();
                                                if (objLGMDPlus.Count == 1)
                                                    objLGMDPlus.RemoveAt(0);
                                                else
                                                    objLGMDPlus.RemoveRange(objLGMDPlus.Count - Npa_lgmdplus_des, Npa_lgmdplus_des);
                                            }
                                            else // only left LGMD TNN agents
                                            {
                                                objLGMDS.Sort();
                                                if (objLGMDS.Count > totalAgents)
                                                    objLGMDS.RemoveRange(totalAgents, objLGMDS.Count - totalAgents);
                                                else
                                                    objLGMDS.RemoveRange(objLGMDS.Count - Npa_lgmdtnn_des, Npa_lgmdtnn_des);
                                            }
                                        }
                                    }
                                    break;
                                }
                            default:
                                break;
                        }
                        Console.WriteLine("Survivor selection: mixed population");
                        break;
                    }
                default:
                    break;
            }
        }

        /// <summary>
        /// Condition of terminating evolution
        /// </summary>
        /// <returns></returns>
        private bool isEvolvingTerminates()
        {
            if (kindGA == 5)    //competitive coevolution
            {
                if (generation == maxGeneration)
                    return true;
                else if (objLGMDS.Count == totalAgents || objLGMDD.Count == totalAgents || ObjLGMDP.Count == totalAgents)
                    return true;
                else
                    return false;
            }
            else if (kindGA == 6)   //run testing data
            {
                return true;
            }
            else
            {
                if (generation == maxGeneration)    //individually evolution (& competitive coevolution)
                    return true;
                else
                    return false;
            }
        }

        /// <summary>
        /// Gaussian perturbation function
        /// </summary>
        /// <param name="change_param"></param>
        /// <param name="bottom_change_param"></param>
        /// <param name="upper_change_param"></param>
        /// <param name="sigma"></param>
        /// <param name="scale"></param>
        /// <returns></returns>
        private float gaussPerturbation(float change_param, float bottom_change_param, float upper_change_param, float sigma, float scale)
        {
            double randDensity, distanceToMean, distanceX, distantToParam;
            float outputParam;
            randDensity = _rand.Next((int)(min_likelihood * 10000), (int)(max_likelihood * 10000)) / 10000.0;
            distanceToMean = gaussianDistanceCalc(randDensity, sigma);
            distanceX = Math.Sqrt(distanceToMean);
            distantToParam = distanceX * change_param / scale;
            if (_rand.Next(0, 10) % 2 == 0)
            {
                outputParam = change_param + (float)distantToParam;
                if (outputParam >= upper_change_param)
                    outputParam = upper_change_param;
            }
            else
            {
                outputParam = change_param - (float)distantToParam;
                if (outputParam <= bottom_change_param)
                    outputParam = bottom_change_param;
            }
            return outputParam;
        }

        /// <summary>
        /// Matching event timing function
        /// </summary>
        /// <param name="detected_timing"></param>
        /// <param name="labelled_start_timing"></param>
        /// <param name="labelled_end_timing"></param>
        /// <param name="allowed_error"></param>
        /// <returns></returns>
        private bool isMatchEventTiming(int detected_timing, int labelled_start_timing, int labelled_end_timing, int allowed_error)
        {
            int compared_start = labelled_start_timing - allowed_error;
            if (compared_start < 1)
                compared_start = 1;
            int compared_end = labelled_start_timing + allowed_error;
            if (compared_end > labelled_end_timing)
                compared_end = labelled_end_timing;
            if (detected_timing >= compared_start && detected_timing <= compared_end)
                return true;
            else
                return false;
        }

        #endregion

        #region EVOLUTION

        /// <summary>
        /// GAs processing
        /// </summary>
        /// <param name="dataPath"></param>
        public void GAs(string dataPath)
        {
            try
            {
                int datasetCount = dataset.Length;
                // initialize population
                if (generation == 1)
                    agentsInitialisation(width, height, fps);
                Console.WriteLine("Current processing generation: {0}", generation);
                // print results to txt file
                using (TextWriter fqbFile = File.CreateText(dataPath + outputTxtFile + generation + ".txt"))
                {
                    // run dataset (all agents)
                    //int datasetCount = dataset.Length;
                    for (int i = 0; i < datasetCount; i++)
                    {
                        string fileName = dataPath + dataset[i].fileName + ".mp4";
                        //string fileName = "c01.mp4";
                        GACapture = new Capture(fileName);
                        frames = (int)GACapture.GetCaptureProperty(CAP_PROP.CV_CAP_PROP_FRAME_COUNT);
                        photos = new List<byte[,,]>();
                        images = new Image<Gray, Byte>[frames];
                        for (int j = 0; j < frames - 1; j++)
                        {
                            GACapture.SetCaptureProperty(CAP_PROP.CV_CAP_PROP_POS_FRAMES, j + 1);
                            images[j] = GACapture.QueryGrayFrame();
                            byte[,,] tmp = images[j].Data;
                            photos.Add(tmp);
                        }

                        // process
                        for (int t = 1; t < frames - 1; t++)
                        {
                            runDataset(photos[t - 1], photos[t], t);
                        }
                        Console.WriteLine();
                        // calculate fitness value of each evolving agent in every training collision and non-collision scenario
                        if (dataset[i].collision == 1) // collision scenario
                        {
                            if (objLGMDPlus.Count > 0) // LGMDPlus model agents exist
                            {
                                foreach (LGMDPlus lgmdP in objLGMDPlus)
                                {
                                    if (!isMatchEventTiming(lgmdP.ActivationTiming, dataset[i].eventStartPoint, dataset[i].eventEndPoint, dataset[i].allowed_error))
                                    {
                                        lgmdP.FCOL++;
                                    }
                                    // reset
                                    lgmdP.COLLISION = 0;
                                    lgmdP.ActivationTiming = 0;
                                }
                            }
                            if (objLGMDD.Count > 0) // LGMD dual channels model agents exist
                            {
                                foreach (LGMDs lgmdD in objLGMDD)
                                {
                                    if (!isMatchEventTiming(lgmdD.ActivationTiming, dataset[i].eventStartPoint, dataset[i].eventEndPoint, dataset[i].allowed_error))
                                    {
                                        lgmdD.FCOL++;
                                    }
                                    // reset
                                    lgmdD.COLLISION = 0;
                                    lgmdD.ActivationTiming = 0;
                                }
                            }
                            if (objLGMDS.Count > 0) // LGMD single pathway model agents exist
                            {
                                foreach (LGMDSingle lgmdS in objLGMDS)
                                {
                                    if (!isMatchEventTiming(lgmdS.ActivationTiming, dataset[i].eventStartPoint, dataset[i].eventEndPoint, dataset[i].allowed_error))
                                    {
                                        lgmdS.FCOL++;
                                    }
                                    // reset
                                    lgmdS.COLLISION = 0;
                                    lgmdS.ActivationTiming = 0;
                                }
                            }
                        }
                        else // non-collision scenario
                        {
                            if (objLGMDPlus.Count > 0) // LGMDPlus model agents exist
                            {
                                foreach (LGMDPlus lgmdP in objLGMDPlus)
                                {
                                    if (lgmdP.COLLISION == 1)
                                    {
                                        lgmdP.FNON++;
                                    }
                                    // reset
                                    lgmdP.COLLISION = 0;
                                    lgmdP.ActivationTiming = 0;
                                }
                            }
                            if (objLGMDD.Count > 0) // LGMD dual channels model agents exist
                            {
                                foreach (LGMDs lgmdD in objLGMDD)
                                {
                                    if (lgmdD.COLLISION == 1)
                                    {
                                        lgmdD.FNON++;
                                    }
                                    // reset
                                    lgmdD.COLLISION = 0;
                                    lgmdD.ActivationTiming = 0;
                                }
                            }
                            if (objLGMDS.Count > 0) // LGMD single pathway model agents exist
                            {
                                foreach (LGMDSingle lgmdS in objLGMDS)
                                {
                                    if (lgmdS.COLLISION == 1)
                                    {
                                        lgmdS.FNON++;
                                    }
                                    // reset
                                    lgmdS.COLLISION = 0;
                                    lgmdS.ActivationTiming = 0;
                                }
                            }
                        }
                        // reset outputs of all populations
                        if (objLGMDPlus.Count > 0)
                        {
                            foreach (LGMDPlus lgmdP in objLGMDPlus)
                            {
                                lgmdP.LGMDPlus_Reset();
                            }
                        }
                        if (objLGMDS.Count > 0)
                        {
                            foreach (LGMDSingle lgmdS in objLGMDS)
                            {
                                lgmdS.LGMDS_Restore();
                            }
                        }
                        if (objLGMDD.Count > 0)
                        {
                            foreach (LGMDs lgmdD in objLGMDD)
                            {
                                lgmdD.LGMDD_Restore();
                            }
                        }
                    }

                    // calculate fitness value after running all data
                    if (objLGMDPlus.Count > 0) // LGMDPlus model agents exist
                    {
                        sumFitness[0] = 0;
                        float tmpMax = 0;
                        int tmpBest = 0;
                        foreach (LGMDPlus lgmdP in objLGMDPlus)
                        {
                            if (!lgmdP.IsOldGeneration)
                            {
                                lgmdP.Fitness = fitnessCalc(lgmdP.FCOL, lgmdP.FNON, lossCol, lossNon);
                                lgmdP.IsOldGeneration = true;
                            }
                            sumFitness[0] += lgmdP.Fitness;
                            if (lgmdP.Fitness >= fitBaseline)
                                tmpBest++;
                            if (tmpMax < lgmdP.Fitness)
                                tmpMax = lgmdP.Fitness;
                            // reset
                            //lgmdP.FCOL = 0;
                            //lgmdP.FNON = 0;
                        }
                        maxFitness[0] = tmpMax;
                        meanFitness[0] = sumFitness[0] / objLGMDPlus.Count;
                        bestAgents[0] = tmpBest;
                    }
                    else
                    {
                        sumFitness[0] = 0;
                        maxFitness[0] = 0;
                        meanFitness[0] = 0;
                        bestAgents[0] = 0;
                    }
                    if (objLGMDD.Count > 0) // LGMD NN model agents exist
                    {
                        sumFitness[2] = 0;
                        float tmpMax = 0;
                        int tmpBest = 0;
                        foreach (LGMDs lgmdD in objLGMDD)
                        {
                            if (!lgmdD.IsOldGeneration)
                            {
                                lgmdD.Fitness = fitnessCalc(lgmdD.FCOL, lgmdD.FNON, lossCol, lossNon);
                                lgmdD.IsOldGeneration = true;
                            }
                            sumFitness[2] += lgmdD.Fitness;
                            if (lgmdD.Fitness >= fitBaseline)
                                tmpBest++;
                            if (tmpMax < lgmdD.Fitness)
                                tmpMax = lgmdD.Fitness;
                            // reset
                            //lgmdD.FCOL = 0;
                            //lgmdD.FNON = 0;
                        }
                        maxFitness[2] = tmpMax;
                        meanFitness[2] = sumFitness[2] / objLGMDD.Count;
                        bestAgents[2] = tmpBest;
                    }
                    else
                    {
                        sumFitness[2] = 0;
                        maxFitness[2] = 0;
                        meanFitness[2] = 0;
                        bestAgents[2] = 0;
                    }
                    if (objLGMDS.Count > 0) // LGMD TNN model agents exist
                    {
                        sumFitness[1] = 0;
                        float tmpMax = 0;
                        int tmpBest = 0;
                        foreach (LGMDSingle lgmdS in objLGMDS)
                        {
                            if (!lgmdS.IsOldGeneration)
                            {
                                lgmdS.Fitness = fitnessCalc(lgmdS.FCOL, lgmdS.FNON, lossCol, lossNon);
                                lgmdS.IsOldGeneration = true;
                            }
                            sumFitness[1] += lgmdS.Fitness;
                            if (lgmdS.Fitness >= fitBaseline)
                                tmpBest++;
                            if (tmpMax < lgmdS.Fitness)
                                tmpMax = lgmdS.Fitness;
                            // reset
                            //lgmdS.FCOL = 0;
                            //lgmdS.FNON = 0;
                        }
                        maxFitness[1] = tmpMax;
                        meanFitness[1] = sumFitness[1] / objLGMDS.Count;
                        bestAgents[1] = tmpBest;
                    }
                    else
                    {
                        sumFitness[1] = 0;
                        maxFitness[1] = 0;
                        meanFitness[1] = 0;
                        bestAgents[1] = 0;
                    }

                    Console.WriteLine("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12}", generation, bestAgents[0], bestAgents[1], bestAgents[2], meanFitness[0], meanFitness[1], meanFitness[2], maxFitness[0], maxFitness[1], maxFitness[2], processLGMDPAgentsNo, processLGMDSAgentsNo, processLGMDDAgentsNo);
                    fqbFile.WriteLine("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12}", generation, bestAgents[0], bestAgents[1], bestAgents[2], meanFitness[0], meanFitness[1], meanFitness[2], maxFitness[0], maxFitness[1], maxFitness[2], processLGMDPAgentsNo, processLGMDSAgentsNo, processLGMDDAgentsNo);

                    // return evolved agents
                   if (objLGMDPlus.Count > 0)
                    {
                        foreach (LGMDPlus lgmdP in objLGMDPlus)
                            fqbFile.WriteLine("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}", 1, lgmdP.Fitness, lgmdP.TFFI, lgmdP.TDE, lgmdP.TSP, lgmdP.TSR, lgmdP.TAU_CEN_E, lgmdP.TAU_HP, lgmdP.W_BASE, lgmdP.COE_SIG, lgmdP.STD_W);
                    }
                    if (objLGMDS.Count > 0)
                    {
                        foreach (LGMDSingle lgmdS in objLGMDS)
                            fqbFile.WriteLine("{0} {1} {2} {3} {4} {5} {6}", 2, lgmdS.Fitness, lgmdS.TFFI, lgmdS.TS, lgmdS.TSP, lgmdS.W_I, lgmdS.COE_SIG);
                    }
                    if (objLGMDD.Count > 0)
                    {
                        foreach (LGMDs lgmdD in objLGMDD)
                            fqbFile.WriteLine("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}", 3, lgmdD.Fitness, lgmdD.TFFI, lgmdD.TS, lgmdD.TSP, lgmdD.COE_SIG, lgmdD.W_I_OFF, lgmdD.W_I_ON, lgmdD.TAU_HP, lgmdD.TAU_LP);
                    }
                }
                // GAs loop
                while (!isEvolvingTerminates()) // termination
                {
                    generation++;
                    using (TextWriter fqbFile = File.CreateText(dataPath + outputTxtFile + generation + ".txt"))
                    {
                        Console.WriteLine("Current processing generation: {0}", generation);

                        // parent selection (fitness based top ranking selection)
                        parentSelection();

                        // crossover (uniform crossover) between pairwise agents with probability Pc
                        Crossover();

                        // mutation (gaussian perturbation) with global probability Pm
                        Mutation();

                        // run dataset (new agents)
                        for (int i = 0; i < datasetCount; i++)
                        {
                            GACapture = new Capture(dataPath + dataset[i].fileName + ".mp4");
                            frames = (int)GACapture.GetCaptureProperty(CAP_PROP.CV_CAP_PROP_FRAME_COUNT);
                            photos = new List<byte[,,]>();
                            images = new Image<Gray, byte>[frames];
                            for (int j = 0; j < frames - 1; j++)
                            {
                                GACapture.SetCaptureProperty(CAP_PROP.CV_CAP_PROP_POS_FRAMES, j + 1);
                                images[j] = GACapture.QueryGrayFrame();
                                byte[,,] tmp = images[j].Data;
                                photos.Add(tmp);
                            }

                            // process (new agents)
                            for (int t = 1; t < frames - 1; t++)
                            {
                                runDataset(photos[t - 1], photos[t], t);
                            }
                            Console.WriteLine();
                            // calculate fitness value of each evolving agent in every training collision and non-collision scenario (new agents)
                            if (dataset[i].collision == 1) // collision scenario
                            {
                                if (objLGMDPlus.Count > 0) // LGMDPlus model agents exist
                                {
                                    foreach (LGMDPlus lgmdP in objLGMDPlus)
                                    {
                                        if (!lgmdP.IsOldGeneration) // new generation
                                        {
                                            if (!isMatchEventTiming(lgmdP.ActivationTiming, dataset[i].eventStartPoint, dataset[i].eventEndPoint, dataset[i].allowed_error))
                                            {
                                                lgmdP.FCOL++;
                                            }
                                            // reset
                                            lgmdP.COLLISION = 0;
                                            lgmdP.ActivationTiming = 0;
                                        }
                                        else
                                            continue;
                                    }
                                }
                                if (objLGMDD.Count > 0) // LGMD dual channels model agents exist
                                {
                                    foreach (LGMDs lgmdD in objLGMDD)
                                    {
                                        if (!lgmdD.IsOldGeneration) // new generation
                                        {
                                            if (!isMatchEventTiming(lgmdD.ActivationTiming, dataset[i].eventStartPoint, dataset[i].eventEndPoint, dataset[i].allowed_error))
                                            {
                                                lgmdD.FCOL++;
                                            }
                                            // reset
                                            lgmdD.COLLISION = 0;
                                            lgmdD.ActivationTiming = 0;
                                        }
                                        else
                                            continue;
                                    }
                                }
                                if (objLGMDS.Count > 0) // LGMD single pathway model agents exist
                                {
                                    foreach (LGMDSingle lgmdS in objLGMDS)
                                    {
                                        if (!lgmdS.IsOldGeneration) // new generation
                                        {
                                            if (!isMatchEventTiming(lgmdS.ActivationTiming, dataset[i].eventStartPoint, dataset[i].eventEndPoint, dataset[i].allowed_error))
                                            {
                                                lgmdS.FCOL++;
                                            }
                                            // reset
                                            lgmdS.COLLISION = 0;
                                            lgmdS.ActivationTiming = 0;
                                        }
                                        else
                                            continue;
                                    }
                                }
                            }
                            else // non-collision scenario
                            {
                                if (objLGMDPlus.Count > 0) // LGMDPlus model agents exist
                                {
                                    foreach (LGMDPlus lgmdP in objLGMDPlus)
                                    {
                                        if (!lgmdP.IsOldGeneration) // new generation
                                        {
                                            if (lgmdP.COLLISION == 1)
                                            {
                                                lgmdP.FNON++;
                                            }
                                            // reset
                                            lgmdP.COLLISION = 0;
                                            lgmdP.ActivationTiming = 0;
                                        }
                                        else
                                            continue;
                                    }
                                }
                                if (objLGMDD.Count > 0) // LGMD dual channels model agents exist
                                {
                                    foreach (LGMDs lgmdD in objLGMDD)
                                    {
                                        if (!lgmdD.IsOldGeneration) // new generation
                                        {
                                            if (lgmdD.COLLISION == 1)
                                            {
                                                lgmdD.FNON++;
                                            }
                                            // reset
                                            lgmdD.COLLISION = 0;
                                            lgmdD.ActivationTiming = 0;
                                        }
                                        else
                                            continue;
                                    }
                                }
                                if (objLGMDS.Count > 0) // LGMD single pathway model agents exist
                                {
                                    foreach (LGMDSingle lgmdS in objLGMDS)
                                    {
                                        if (!lgmdS.IsOldGeneration) //new generation
                                        {
                                            if (lgmdS.COLLISION == 1)
                                            {
                                                lgmdS.FNON++;
                                            }
                                            // reset
                                            lgmdS.COLLISION = 0;
                                            lgmdS.ActivationTiming = 0;
                                        }
                                        else
                                            continue;
                                    }
                                }
                            }
                            // reset outputs of all new population
                            if (objLGMDPlus.Count > 0)
                            {
                                foreach (LGMDPlus lgmdP in objLGMDPlus)
                                {
                                    if (!lgmdP.IsOldGeneration)
                                        lgmdP.LGMDPlus_Reset();
                                }
                            }
                            if (objLGMDS.Count > 0)
                            {
                                foreach (LGMDSingle lgmdS in objLGMDS)
                                {
                                    if (!lgmdS.IsOldGeneration)
                                        lgmdS.LGMDS_Restore();
                                }
                            }
                            if (objLGMDD.Count > 0)
                            {
                                foreach (LGMDs lgmdD in objLGMDD)
                                {
                                    if (!lgmdD.IsOldGeneration)
                                        lgmdD.LGMDD_Restore();
                                }
                            }
                        }

                        // calculate fitness value after running all data (new agents)
                        if (objLGMDPlus.Count > 0) // LGMDPlus model agents exist
                        {
                            sumFitness[0] = 0;
                            float tmpMax = 0;
                            int tmpBest = 0;
                            foreach (LGMDPlus lgmdP in objLGMDPlus)
                            {
                                if (!lgmdP.IsOldGeneration)
                                {
                                    lgmdP.Fitness = fitnessCalc(lgmdP.FCOL, lgmdP.FNON, lossCol, lossNon);
                                    lgmdP.IsOldGeneration = true;
                                }
                                sumFitness[0] += lgmdP.Fitness;
                                if (lgmdP.Fitness >= fitBaseline)
                                    tmpBest++;
                                if (tmpMax < lgmdP.Fitness)
                                    tmpMax = lgmdP.Fitness;
                                // reset
                                //lgmdP.FCOL = 0;
                                //lgmdP.FNON = 0;
                            }
                            maxFitness[0] = tmpMax;
                            meanFitness[0] = sumFitness[0] / objLGMDPlus.Count;
                            bestAgents[0] = tmpBest;
                        }
                        else
                        {
                            sumFitness[0] = 0;
                            maxFitness[0] = 0;
                            meanFitness[0] = 0;
                            bestAgents[0] = 0;
                        }
                        if (objLGMDD.Count > 0) // LGMD NN model agents exist
                        {
                            sumFitness[2] = 0;
                            float tmpMax = 0;
                            int tmpBest = 0;
                            foreach (LGMDs lgmdD in objLGMDD)
                            {
                                if (!lgmdD.IsOldGeneration)
                                {
                                    lgmdD.Fitness = fitnessCalc(lgmdD.FCOL, lgmdD.FNON, lossCol, lossNon);
                                    lgmdD.IsOldGeneration = true;
                                }
                                sumFitness[2] += lgmdD.Fitness;
                                if (lgmdD.Fitness >= fitBaseline)
                                    tmpBest++;
                                if (tmpMax < lgmdD.Fitness)
                                    tmpMax = lgmdD.Fitness;
                                // reset
                                //lgmdD.FCOL = 0;
                                //lgmdD.FNON = 0;
                            }
                            maxFitness[2] = tmpMax;
                            meanFitness[2] = sumFitness[2] / objLGMDD.Count;
                            bestAgents[2] = tmpBest;
                        }
                        else
                        {
                            sumFitness[2] = 0;
                            maxFitness[2] = 0;
                            meanFitness[2] = 0;
                            bestAgents[2] = 0;
                        }
                        if (objLGMDS.Count > 0) // LGMD TNN model agents exist
                        {
                            sumFitness[1] = 0;
                            float tmpMax = 0;
                            int tmpBest = 0;
                            foreach (LGMDSingle lgmdS in objLGMDS)
                            {
                                if (!lgmdS.IsOldGeneration)
                                {
                                    lgmdS.Fitness = fitnessCalc(lgmdS.FCOL, lgmdS.FNON, lossCol, lossNon);
                                    lgmdS.IsOldGeneration = true;
                                }
                                sumFitness[1] += lgmdS.Fitness;
                                if (lgmdS.Fitness >= fitBaseline)
                                    tmpBest++;
                                if (tmpMax < lgmdS.Fitness)
                                    tmpMax = lgmdS.Fitness;
                                // reset
                                //lgmdS.FCOL = 0;
                                //lgmdS.FNON = 0;
                            }
                            maxFitness[1] = tmpMax;
                            meanFitness[1] = sumFitness[1] / objLGMDS.Count;
                            bestAgents[1] = tmpBest;
                        }
                        else
                        {
                            sumFitness[1] = 0;
                            maxFitness[1] = 0;
                            meanFitness[1] = 0;
                            bestAgents[1] = 0;
                        }

                        // fitness based selection of survivals
                        survivorSelection();

                        // updating processing agents numbers of different model populations
                        processLGMDPAgentsNo = objLGMDPlus.Count;
                        processLGMDDAgentsNo = objLGMDD.Count;
                        processLGMDSAgentsNo = objLGMDS.Count;

                        Console.WriteLine("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12}", generation, bestAgents[0], bestAgents[1], bestAgents[2], meanFitness[0], meanFitness[1], meanFitness[2], maxFitness[0], maxFitness[1], maxFitness[2], processLGMDPAgentsNo, processLGMDSAgentsNo, processLGMDDAgentsNo);
                        fqbFile.WriteLine("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12}", generation, bestAgents[0], bestAgents[1], bestAgents[2], meanFitness[0], meanFitness[1], meanFitness[2], maxFitness[0], maxFitness[1], maxFitness[2], processLGMDPAgentsNo, processLGMDSAgentsNo, processLGMDDAgentsNo);
                        
                        // return evolved agents
                        if (objLGMDPlus.Count > 0)
                        {
                            foreach (LGMDPlus lgmdP in objLGMDPlus)
                                fqbFile.WriteLine("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}", 1, lgmdP.Fitness, lgmdP.TFFI, lgmdP.TDE, lgmdP.TSP, lgmdP.TSR, lgmdP.TAU_CEN_E, lgmdP.TAU_HP, lgmdP.W_BASE, lgmdP.COE_SIG, lgmdP.STD_W);
                        }
                        if (objLGMDS.Count > 0)
                        {
                            foreach (LGMDSingle lgmdS in objLGMDS)
                                fqbFile.WriteLine("{0} {1} {2} {3} {4} {5} {6}", 2, lgmdS.Fitness, lgmdS.TFFI, lgmdS.TS, lgmdS.TSP, lgmdS.W_I, lgmdS.COE_SIG);
                        }
                        if (objLGMDD.Count > 0)
                        {
                            foreach (LGMDs lgmdD in objLGMDD)
                                fqbFile.WriteLine("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}", 3, lgmdD.Fitness, lgmdD.TFFI, lgmdD.TS, lgmdD.TSP, lgmdD.COE_SIG, lgmdD.W_I_OFF, lgmdD.W_I_ON, lgmdD.TAU_HP, lgmdD.TAU_LP);
                        }
                    }
                }
                //Console.WriteLine();
                //Console.ReadKey();
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
        }

        /// <summary>
        /// GAs processing continues from interruption
        /// </summary>
        /// <param name="gaProcessTxtFile"></param>
        /// <returns></returns>
        public int GAs_Continue(string gaProcessTxtFile)
        {
            int num = 1;
            using (StreamReader rf = new StreamReader(gaProcessTxtFile))
            {
                if (rf == null)
                {
                    Console.WriteLine("Label file not found...\n");
                    return 0;
                }
                string s;
                while ((s = rf.ReadLine()) != null)
                {
                    Console.WriteLine(s);
                    //Split each line by 'space'
                    string[] split = s.Split(' ');
                    if (num == 1)
                    {
                        generation = int.Parse(split[0]) + 1;
                    }
                    else
                    {
                        if (kindGA == 1)    // LGMDPlus model agents
                        {
                            LGMDPlus lgmdP = new LGMDPlus(width, height, fps);
                            lgmdP.ParamsDict["Tffi"] = int.Parse(split[2]);
                            lgmdP.ParamsDict["Tde"] = int.Parse(split[3]);
                            lgmdP.ParamsDict["Tsp"] = float.Parse(split[4]);
                            lgmdP.ParamsDict["Tsr"] = int.Parse(split[5]);
                            lgmdP.ParamsDict["tau_E"] = float.Parse(split[6]);
                            lgmdP.ParamsDict["tau_hp"] = int.Parse(split[7]);
                            lgmdP.ParamsDict["W_base"] = float.Parse(split[8]);
                            lgmdP.ParamsDict["coe_sig"] = float.Parse(split[9]);
                            lgmdP.ParamsDict["std_w"] = float.Parse(split[10]);
                            lgmdP.LGMDPlus_searchingParametersExchanging();
                            objLGMDPlus.Add(lgmdP);
                        }
                        else if (kindGA == 2)   // LGMD TNN model agents
                        {
                            LGMDSingle lgmdS = new LGMDSingle(width, height);
                            lgmdS.ParamsDict["Tffi"] = int.Parse(split[2]);
                            lgmdS.ParamsDict["Ts"] = int.Parse(split[3]);
                            lgmdS.ParamsDict["Tsp"] = float.Parse(split[4]);
                            lgmdS.ParamsDict["Wi"] = float.Parse(split[5]);
                            lgmdS.ParamsDict["coe_sig"] = float.Parse(split[6]);
                            lgmdS.LGMDS_searchingParametersExchanging();
                            objLGMDS.Add(lgmdS);
                        }
                        else if (kindGA == 3)   // LGMD NN model agents
                        {
                            LGMDs lgmdD = new LGMDs(width, height, fps);
                            lgmdD.ParamsDict["Tffi"] = int.Parse(split[2]);
                            lgmdD.ParamsDict["Ts"] = int.Parse(split[3]);
                            lgmdD.ParamsDict["Tsp"] = float.Parse(split[4]);
                            lgmdD.ParamsDict["coe_sig"] = float.Parse(split[5]);
                            lgmdD.ParamsDict["W_i_off"] = float.Parse(split[6]);
                            lgmdD.ParamsDict["W_i_on"] = float.Parse(split[7]);
                            lgmdD.ParamsDict["tau_hp"] = int.Parse(split[8]);
                            lgmdD.ParamsDict["tau_lp"] = int.Parse(split[9]);
                            lgmdD.LGMDD_searchingParametersExchanging();
                            objLGMDD.Add(lgmdD);
                        }
                    }
                    
                    num++;
                }
            }
            return 1;   //success
        }

        /// <summary>
        /// Run piece of dataset
        /// </summary>
        /// <param name="inputDataPath"></param>
        /// <param name="processVideoFile"></param>
        /// <returns></returns>
        public int Run_Piece_Of_Data(string inputDataPath, string processVideoFile)
        {
            GACapture = new Capture(processVideoFile);
            frames = (int)GACapture.GetCaptureProperty(CAP_PROP.CV_CAP_PROP_FRAME_COUNT);
            photos = new List<byte[,,]>();
            images = new Image<Gray, Byte>[frames];
            for (int j = 0; j < frames - 1; j++)
            {
                GACapture.SetCaptureProperty(CAP_PROP.CV_CAP_PROP_POS_FRAMES, j + 1);
                images[j] = GACapture.QueryGrayFrame();
                byte[,,] tmp = images[j].Data;
                photos.Add(tmp);
            }
            // process
            if (objLGMDPlus.Count > 0)  // LGMDPlus models processing
            {
                int num = 1;
                foreach (LGMDPlus lgmdP in objLGMDPlus)
                {
                    using (TextWriter fqbFile = File.CreateText(inputDataPath + "obj" + num + ".txt"))
                    {
                        for (int t = 1; t < frames - 1; t++)
                        {
                            int cur_frame = t % lgmdP.ONs.GetLength(2);
                            int cur_spi = t % lgmdP.SPIKE.Length;
                            lgmdP.LGMDPlus_Processing(photos[t - 1], photos[t], t);
                            fqbFile.WriteLine("{0} {1:F} {2:F} {3} {4:F} {5:F} {6:F}", t, lgmdP.SMP[cur_frame], lgmdP.SFA[cur_frame], lgmdP.SPIKE[cur_spi], lgmdP.SPIKERATE, lgmdP.FFI[cur_frame], lgmdP.DYN_TAU_G);
                        }
                    }
                    num++;
                }
            }
            else
                return 0;

            return 1;   //success
        }

        /// <summary>
        /// Run all data and output model response
        /// </summary>
        /// <param name="inputDataPath"></param>
        /// <returns></returns>
        public int Run_All_Pieces_Of_Data(string inputDataPath)
        {
            string fqbDir = @"ga-lgmd-plus-data-piece\20agents-2\";
            int datasetCount = dataset.Length;
            // process
            if (objLGMDPlus.Count > 0)  // LGMDPlus models processinng
            {
                for (int i = 0; i < datasetCount; i++)
                {
                    // create directory
                    string dirPath = inputDataPath + fqbDir + dataset[i].fileName;
                    if (Directory.Exists(dirPath))
                    {
                        Console.WriteLine("That path has existed already.");
                        return 0;
                    }
                    DirectoryInfo di = Directory.CreateDirectory(dirPath);
                    Console.WriteLine("The directory was created successfully at {0}.", Directory.GetCreationTime(dirPath));

                    // process input video
                    string videoFileName = inputDataPath + dataset[i].fileName + ".mp4";
                    GACapture = new Capture(videoFileName);
                    frames = (int)GACapture.GetCaptureProperty(CAP_PROP.CV_CAP_PROP_FRAME_COUNT);
                    photos = new List<byte[,,]>();
                    images = new Image<Gray, Byte>[frames];
                    for (int j = 0; j < frames - 1; j++)
                    {
                        GACapture.SetCaptureProperty(CAP_PROP.CV_CAP_PROP_POS_FRAMES, j + 1);
                        images[j] = GACapture.QueryGrayFrame();
                        byte[,,] tmp = images[j].Data;
                        photos.Add(tmp);
                    }

                    int num = 1;
                    foreach (LGMDPlus lgmdP in objLGMDPlus)
                    {
                        using (TextWriter fqbFile = File.CreateText(dirPath + @"\obj" + num + ".txt"))
                        {
                            for (int t = 1; t < frames - 1; t++)
                            {
                                int cur_frame = t % lgmdP.ONs.GetLength(2);
                                int cur_spi = t % lgmdP.SPIKE.Length;
                                lgmdP.LGMDPlus_Processing(photos[t - 1], photos[t], t);
                                fqbFile.WriteLine("{0} {1:F} {2:F} {3} {4:F} {5:F} {6:F}", t, lgmdP.SMP[cur_frame], lgmdP.SFA[cur_frame], lgmdP.SPIKE[cur_spi], lgmdP.SPIKERATE, lgmdP.FFI[cur_frame], lgmdP.DYN_TAU_G);
                            }
                        }
                        num++;
                    }
                }
            }
            else
                return 0;

            return 1;   //success
        }

        #endregion
    }
}

