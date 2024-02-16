clear all
%close all
clc
% run("cleanup.m")
addpath(genpath(pwd));
% ray-tracing environment size and granularity
% super C  x: 350, y: 350, granularity 1
% frankfurt  x: 750, y: 750, granularity 1
x_size = 750;
y_size = 750;
granularity = 1;

% pencil beam paths

% rx_number_of_beams = 4;
% rx_step_size = 360 / (rx_number_of_beams * 4);
% rx_path = "4x4_" + int2str(rx_number_of_beams) + "beams_" + num2str(rx_step_size) + "step";
rx_number_of_beams = 4;
tx_number_of_beams = 16;

% get TX ID (1 or 2) so that correct ray-tracing file will be opened


% transmit power levels and rx_sensitivity
P_TX = 16;
RX_Sen = -68;

% TX_angleName = "TxRx_Angles.mat";
% % /home/panwei/Desktop/summer23/masterArbeit/rss_evaluation_toolbox_superC/results/Frankfurt_28GHz/5194466/0p1_new.mat
% TX_resultName = "0p1_new.mat";
% % savePathPre = "bpl_data/frankfurt/";
% savePathPre = "bpl_data/superC/";
% % matSavePathPre = "bpl_data_mat/frankfurt/";
% matSavePathPre = "bpl_data_mat/superC/";
% % resultPathParent = "./results/Frankfurt_28GHz";
% resultPathParent = "./results/result_TX_1_16dBm_OE";

TX_angleName = "TxRx_Angles.mat";
% /home/panwei/Desktop/summer23/masterArbeit/rss_evaluation_toolbox_superC/results/Frankfurt_28GHz/5194466/0p1_new.mat
TX_resultName = "0p1_new.mat";
savePathPre = "bpl_data/frankfurt/";
matSavePathPre = "bpl_data_mat/frankfurt/";
resultPathParent = "./results/Frankfurt_28GHz";

% chooseLOS = true;
chooseLOS = false;
chooseOnlyNLOS = false;
rxOmni = false;

% saved final csv folder path
dataSavePathPre = "data_frankfurt"

% TX_stationLists = [5194466];
% TX_stationLists = [1];
% TX_stationLists = ["5194466",
TX_stationLists = [
%     5194466,
% 3054566,
% 5836116,
% 4434416,
% 4581536,
% 3163526,
% 5853136,
3625286,
% 5955386,
3552206,
3725856,
4063746,
1193486,
5751526,
1564986,
5355566,
5203856,
1625746,
4465066,
4513396,
2805226,
3251696,
182996,
3821656,
4272166,
3594326,
5285056,
5942206,
3782306,
2376016,
5152246,
5101656,
5392746,
5853826,
3453006,
4362816,
2484006,
2245166,
1893956,
6064536,
1774496,
2174486,
4475786,
3085836,
2261176,
3623066
]


% loop through all tx
for txIndex = 1 : length(TX_stationLists)
    TX_station = TX_stationLists(txIndex);
    % get bpl matrix
    % calculate_bpl_matrix(x_size, y_size, rx_number_of_beams,tx_number_of_beams, resultPathParent, TX_station, TX_angleName, TX_resultName, savePathPre, chooseLOS, chooseOnlyNLOS, rxOmni, P_TX, RX_Sen);

    % combine the matrix to single one
    % combineSingleMat(x_size,y_size,granularity, rx_number_of_beams,tx_number_of_beams, TX_station, savePathPre, matSavePathPre, chooseLOS,chooseOnlyNLOS, rxOmni);
    generateDataCSV(x_size,y_size,granularity,rx_number_of_beams,tx_number_of_beams,TX_station, savePathPre, dataSavePathPre, chooseLOS,chooseOnlyNLOS, rxOmni);

end


function calculate_bpl_matrix(x_size, y_size, rx_number_of_beams, tx_number_of_beams, resultPathParent,TX_station, TX_angleName, TX_resultName,  savePathPre, chooseLOS,chooseOnlyNLOS, rxOmni, P_TX, RX_Sen)

    Rx_AziAngle = [];
    Rx_EleAngle = [];
    Tx_AziAngle = [];
    Tx_EleAngle = [];
    Receiver_Ray = [];
    rx_phis = [];
    tx_phis = [];
    rx_ids = [];
    tx_ids = [];
    rx_tt_angs = [];
    tx_tt_angs = [];
    rx_step_size = 360 / (rx_number_of_beams * 4);
    % tx_path = "4x4_" + int2str(rx_number_of_beams) + "beams_" + num2str(rx_step_size) + "step";
    rx_path = int2str(rx_number_of_beams) + "x" + int2str(rx_number_of_beams) + "_" + int2str(rx_number_of_beams) + "beams_" + num2str(rx_step_size) + 'step';


    tx_step_size = 360 / (tx_number_of_beams * 4);
    % tx_path = "4x4_" + int2str(tx_number_of_beams) + "beams_" + num2str(tx_step_size) + "step";
    if tx_number_of_beams ~= 16
        tx_path = int2str(tx_number_of_beams) + "x" + int2str(tx_number_of_beams) + "_" + int2str(tx_number_of_beams) + "beams_" + num2str(tx_step_size) + 'step';
    else
        tx_path = "8x8_" + int2str(tx_number_of_beams)+ "beams_" + num2str(tx_step_size) + 'step';
    end
    % tx_path = "8x8_" + int2str(tx_number_of_beams) + "beams_" + num2str(tx_step_size) + "step";
    
    % save_path= ["350x350_data_OE_TX" num2str(TX_station) "_"];
    save_path= num2str(x_size) + "x" + num2str(y_size) + "_data_OE_TX" +num2str(TX_station) + "_" + "rx" + num2str(rx_number_of_beams) + "tx" + num2str(tx_number_of_beams) + "_";
    oe_directory = save_path;
    % oe_directory = "350x350_data_OE_TX" + num2str(TX_station) + "_" + "rx" + num2str(rx_number_of_beams) + "tx" + num2str(tx_number_of_beams) + "_";
    if chooseLOS
        save_path = save_path + "LOS";
        oe_directory = oe_directory + "LOS";
    elseif chooseOnlyNLOS
        save_path = save_path + "onlyNLOS";
        oe_directory = oe_directory + "onlyNLOS";
    end
    
    if rxOmni
        save_path = save_path + "rxOmni";
        oe_directory = oe_directory + "rxOmni";
    end
    
    % build the path
    save_path = fullfile(savePathPre, save_path);
    oe_directory = fullfile(savePathPre, oe_directory);
    % save_path = save_path + "/";
    % oe_directory = oe_directory + "/";
    mkdir(save_path)
    % result_matrix
    output_matrix = cell(x_size, y_size);
    
    
    % specify framework and antenna pattern parameters
    new_framework = true; % false for old framework
    antenna_pattern = "3D_pattern"; % 3D_pattern or 3D_pattern_simulated



    % RX angle 0� tuning
    Semi90_coordinates = [188.170, 134.528];
    SuperC_coordinates = [123.908, 139.956];

    %% get ray-tracing data
    % gets RX_AziAngle,RX_EleAngle,TX_AziAngle,TX_EleAngle
    % gets Receiver_Ray
    display('Accessing Ray-Tracing Data.')
    if TX_station==1
        load('./results/result_TX_1_16dBm_OE/TxRx_Angles.mat');
        load('./results/result_TX_1_16dBm_OE/SimulationRecord.mat');

        TX_coordinates = [169,128];
    elseif TX_station == 2
        % load either the 8dBm, 16dBm or 24 dBm data
        P_TX_string = num2str(P_TX);
        load(['./results/result_TX_2_' P_TX_string 'dBm_OE/TxRx_Angles.mat']);
        load(['./results/result_TX_2_' P_TX_string 'dBm_OE/SimulationRecord.mat']);

        TX_coordinates = [174,126];
    else
        load(fullfile(resultPathParent, num2str(num2str(TX_station)), TX_angleName));
        load(fullfile(resultPathParent, num2str(num2str(TX_station)), TX_resultName));
    end
    clearvars Num_Azi Num_Ele
    clearvars Azimuth_Stepsize Elevation_Stepsize
    clearvars Rec_Id Rx_TotalPower_dBm_Matrix
    clearvars P_TX_string

    %% get pencil beam patterns from measurements
    % consider pencil beam gains and consider gains

    % load pencil beam data RX side
    % load(['beam_patterns_RX_pencil/' rx_path '/measrun.mat'])
    load('beam_patterns_RX_pencil/' + rx_path + '/measrun.mat')
    if rxOmni
        % for i = 1 : length(A)
        %     A_rx{i} = ones(size(A{i}));
        % end
        A_rx = {};
        A_rx{1} = ones(size(A{1}));
    else
        A_rx = A;
    end

    clearvars A
    rx_beam = ["pencil"];
    % load pencil beam data TX side
    % load(['beam_patterns_TX_pencil/' tx_path '/measrun.mat'])
    load('beam_patterns_TX_pencil/' + tx_path + '/measrun.mat')
    A_tx = A;
    tx_beam = ["pencil"];
    clearvars A

    % the simulated pattern already includes the 16 dB gain
    antenna_gains = 2*0; % dB
    %antenna_gains = 2*16; % dB

    
    %% preparing data for beamsteering algorithms
    
    % create folder
    % mkdir(['bpl_data/' save_path])
    % mkdir('bpl_data/'+ save_path)
    processedDataCount = 0;
    saveMatFileNames=[];
    parpool(20);
    parfor x_pos = 1:x_size
    % for x_pos = 1:x_size
        for y_pos = 1:y_size

            % display(['RX position: ', num2str(x_pos), ', ', num2str(y_pos)])
            % get power
            transmit_power_difference = P_TX-16; % current transmit power difference in dB
            % get RX position and correct RX az. angle
            RX_coordinates = [x_pos, y_pos];

            %% select ray-tracing data for current position
            relevant_ray = Receiver_Ray(RX_coordinates(1),RX_coordinates(2));
            processData = false;
            % disp(relevant_ray.LOS_NLOS)
            % disp(relevant_ray.LOS_NLOS)
            if ~isempty(relevant_ray.Power_dBm)
                % select only LOS path
                if (chooseLOS) && (length(relevant_ray.LOS_NLOS) >= 1) && (relevant_ray.LOS_NLOS(1) == 1)
                    % only LOS path
                    processData = true;
                elseif (chooseOnlyNLOS) && (length(relevant_ray.LOS_NLOS) >= 1) && (relevant_ray.LOS_NLOS(1) ~= 1)
                    % only choose NLOS path
                    processData = true;
                elseif (~chooseLOS) && (~chooseOnlyNLOS)
                    processData = true;
                end
            
            end

            if processData
                relevant_ray.Power_dBm = relevant_ray.Power_dBm + antenna_gains;
                relevant_ray.TotalPower_dBm = relevant_ray.TotalPower_dBm + antenna_gains;
                relevant_ray.TotalPower_mW = 10^(relevant_ray.TotalPower_dBm/10);
                processedDataCount = processedDataCount + 1;
            else
                % no received signal for this position, skip further execution
                % display('No data for this RX position.')
                % meas_tuple = struct();
                % meas_tuple.rss = [];
                % meas_tuple.rx_sensitivity = [];
                % meas_tuple.p_tx = [];
                % meas_tuple.phi_rx = [];
                % meas_tuple.rx_tt_ang = [];
                % meas_tuple.rx_beam_id = [];
                % meas_tuple.rx_beam_type = [];
                % meas_tuple.theta_rx = [];
                % meas_tuple.phi_tx = [];
                % meas_tuple.tx_tt_ang = [];
                % meas_tuple.tx_beam_id = [];
                % meas_tuple.tx_beam_type = [];
                % meas_tuple.theta_tx = [];
                % meas_tuple.is_valid = 0;

                % % folder_path = ['bpl_data/' save_path num2str(RX_coordinates(1)) "_" num2str(RX_coordinates(2)) '.mat'];

                % folder_path = fullfile(save_path, num2str(RX_coordinates(1)) + "_" + num2str(RX_coordinates(2)) + '.mat');
                % parsave(folder_path,meas_tuple)
                % disp("saved at folder:"+ save_path)
                % %output_matrix{RX_coordinates(1),RX_coordinates(2)} = meas_tuple;
                %clearvars folder_path meas_tuple

                continue;
            end

            rx_az_angs = squeeze(Rx_AziAngle(RX_coordinates(1),RX_coordinates(2),:));
            rx_el_angs = squeeze(Rx_EleAngle(RX_coordinates(1),RX_coordinates(2),:));
            tx_az_angs = squeeze(Tx_AziAngle(RX_coordinates(1),RX_coordinates(2),:));
            tx_el_angs = squeeze(Tx_EleAngle(RX_coordinates(1),RX_coordinates(2),:));

            % select correct RSS values depending on LOS/NLOS-ness of ray
            rx_az_angs = rx_az_angs(1:length(relevant_ray.Power_dBm));
            tx_az_angs = tx_az_angs(1:length(relevant_ray.Power_dBm));
            rx_el_angs = rx_el_angs(1:length(relevant_ray.Power_dBm));
            tx_el_angs = tx_el_angs(1:length(relevant_ray.Power_dBm));

            %% create measurement data based on ray-tracing emulating SuperC campaign
            % display('Emulating phased-antenna array.')
            iter = 0;
            % bar_iter = waitbar(0,'Emulation process.');
            rss = [];
            rx_sensitivity = [];
            p_tx = [];
            phi_rx = [];
            rx_tt_ang = [];
            rx_beam_id = [];
            rx_beam_type = [];
            theta_rx = [];
            phi_tx = [];
            tx_tt_ang = [];
            tx_beam_id = [];
            tx_beam_type = [];
            theta_tx = [];
            is_valid = [];
            for m = 1:size(A_tx,1)
                for n = 1:size(A_rx,2)
                    for theta = [-30,0,30]
                        
                        if ~new_framework
                            th_shift_rx = rx_phis(n) + 135; % add 135 as A_rx{24} has turntable at 135 
                            A_rx_tmp = A_rx{24};
                            A_rx_tmp = circshift(A_rx_tmp, th_shift_rx, 2);
                            A_rx_old = {A_rx_tmp};
                            %clearvars A_rx_tmp th_shift_rx

                            th_shift_tx = tx_phis(m);
                            A_tx_old = circshift(A_tx(24,:), th_shift_tx, 2);

                            [RSS,validity] = get_beam_pair_RSS(A_tx_old, A_rx_old,...
                                                tx_az_angs, rx_az_angs,...
                                                tx_el_angs, rx_el_angs,...
                                                relevant_ray.Power_dBm, RX_Sen,...
                                                theta);
                        else
                            tx_az_angs1 = tx_az_angs - 181;
                            rx_az_angs1 = rx_az_angs - 181;
                            [RSS,validity] = get_beam_pair_RSS(A_tx(m,:), A_rx(n),...
                                                tx_az_angs1, rx_az_angs1,...
                                                tx_el_angs, rx_el_angs,...
                                                relevant_ray.Power_dBm, RX_Sen,...
                                                theta);

                        end

                        rss = [rss,RSS];
                        rx_sensitivity = [rx_sensitivity,RX_Sen];
                        p_tx = [p_tx, P_TX];
                        phi_rx = [phi_rx,rx_phis(n)];
                        rx_tt_ang = [rx_tt_ang,rx_tt_angs(n)];
                        rx_beam_id = [rx_beam_id,rx_ids(n)];
                        rx_beam_type = [rx_beam_type,rx_beam];
                        theta_rx = [theta_rx,theta];
                        phi_tx = [phi_tx,tx_phis(m)];
                        tx_tt_ang = [tx_tt_ang,tx_tt_angs(m)];
                        tx_beam_id = [tx_beam_id,tx_ids(m)];
                        tx_beam_type = [tx_beam_type,tx_beam];
                        theta_tx = [theta_tx,-10];
                        is_valid = [is_valid, validity];
                        %clearvars RSS validity
                        iter = iter + 1;
                        % waitbar(iter/(size(A_tx,1)*size(A_rx,2)*3),bar_iter,'Emulation process.')
                    end
                    %clearvars theta
                end
                %clearvars n
            end
            % close(bar_iter)
            %clearvars m iter bar_iter
            %clearvars relevant_ray
            %clearvars rx_az_angs tx_az_angs rx_el_angs tx_el_angs

            %% create and save meas_tuple
            % display(['Saving data for RX position: ', num2str(x_pos), ', ', num2str(y_pos)])
            meas_tuple = struct();
            meas_tuple.rss = rss;
            meas_tuple.rx_sensitivity = rx_sensitivity;
            meas_tuple.p_tx = p_tx;
            meas_tuple.phi_rx = phi_rx;
            meas_tuple.rx_tt_ang = rx_tt_ang;
            meas_tuple.rx_beam_id = rx_beam_id;
            meas_tuple.rx_beam_type = rx_beam_type;
            meas_tuple.theta_rx = theta_rx;
            meas_tuple.phi_tx = phi_tx;
            meas_tuple.tx_tt_ang = tx_tt_ang;
            meas_tuple.tx_beam_id = tx_beam_id;
            meas_tuple.tx_beam_type = tx_beam_type;
            meas_tuple.theta_tx = theta_tx;
            meas_tuple.is_valid = is_valid;
            %clearvars rss rx_sensitivity p_tx is_valid
            %clearvars phi_rx rx_tt_ang rx_beam_id rx_beam_type theta_rx
            %clearvars phi_tx tx_tt_ang tx_beam_id tx_beam_type theta_tx

            % folder_path = ['bpl_data/' oe_directory num2str(RX_coordinates(1)) "_" num2str(RX_coordinates(2)) '.mat'];
            folder_path = fullfile(oe_directory, num2str(RX_coordinates(1)) +"_" + num2str(RX_coordinates(2)) + '.mat');
            tempStr = num2str(RX_coordinates(1)) +"_" + num2str(RX_coordinates(2)) + '.mat'
            saveMatFileNames=[saveMatFileNames; tempStr];
            % break;
            parsave(folder_path,meas_tuple)
            %output_matrix{RX_coordinates(1),RX_coordinates(2)} = meas_tuple;
            %clearvars folder_path meas_tuple
            
        end
    end

    writematrix([processedDataCount],fullfile(oe_directory,"processedData.txt"));

    % writematrix()
    % t means if the file exists, content is destroyed
    fid=fopen(fullfile(oe_directory, "matName.txt"), "wt");
    fprintf(fid, "%s\n",saveMatFileNames(:));
    fclose(fid);

    % save(fullfile(folder_path, "matName.txt"), saveMatFileNames);
    %folder_path = ['emulated_measurements_pencil_beam_pairs/output_matrix.mat'];
    %save(folder_path,'output_matrix')
end

function generateDataCSV(x_size,y_size,granularity,rx_number_of_beams,tx_number_of_beams,TX_station, savePathPre, dataSavePathPre, chooseLOS,chooseOnlyNLOS, rxOmni)

    meas_tuple = [];

    %% RX antenna parameters
    % 4x4 4 beams 22.5 steps
    rx_stepSize = 360 / (4 * rx_number_of_beams);
    rx_phi = -180:rx_stepSize:(180-1);

    %% TX antenna parameters
    % 8x8 16 beams 5.625 steps  16 = 64 / 4   4 quadrants
    tx_stepSize = 360 / (4 * tx_number_of_beams);
    tx_phi = -180:tx_stepSize:(180-1);

    % bpl_table = cell([x_size,y_size]);

    % tx_data_file="350x350_data_OE_TX1/";

    mat_save_name = "rx_" + num2str(rx_number_of_beams) + "tx_" + num2str(tx_number_of_beams);

    % save_path= ["350x350_data_OE_TX" num2str(TX_station) "_"]
    % save_path= "350x350_data_OE_TX" + num2str(TX_station) + "_";
    % save_path= "35                                        0x350_data_OE_TX" +num2str(TX_station) + "_" + "rx" + num2str(rx_number_of_beams) + "tx" + num2str(tx_number_of_beams) + "_";

    save_path= num2str(x_size) + "x" + num2str(y_size) + "_data_OE_TX" +num2str(TX_station) + "_" + "rx" + num2str(rx_number_of_beams) + "tx" + num2str(tx_number_of_beams) + "_";
    oe_directory = save_path;
    if chooseLOS
        save_path = save_path + "LOS";
        mat_save_name = mat_save_name + "_LOS";
    elseif chooseOnlyNLOS
        save_path = save_path + "onlyNLOS";
        oe_directory = oe_directory + "onlyNLOS";      
        mat_save_name = mat_save_name + "_onlyNLOS";
    end
    if rxOmni
        save_path = save_path + "rxOmni";
        % oe_directory = oe_directory + "rxOmni";
        mat_save_name = mat_save_name + "_rxOmni";
        rx_phi = 0;
    end

    save_path = fullfile(savePathPre, save_path);
    % mat_save_name = mat_save_name + ".mat";
    % make dir 
    % data_frankfurt/TXxxxxx/
    data_save_path = fullfile(dataSavePathPre, "TX"+num2str(TX_station));

    mkdir(data_save_path)
    % mat_save_name = fullfile(dataSavePathPre, "TX"+ num2str(TX_station), mat_save_name)

    % get the filePathName
    saveMatFileNames = fullfile(save_path,"matName.txt");
    if isfile(saveMatFileNames)
        saveMatFiles = readlines(saveMatFileNames,"EmptyLineRule","skip");
        readFileName = true;
    else
        readFileName = false;
    end



    % dataMatPrePath = "frankfurt";
    myConfig = load_config("config.ini");
    dataSaveFilePrefix = myConfig("dataSaveFilePrefix");

    % load(fullfile(dataMatPrePath, dataMatFolder, myConfig("dataMatName")))

    % data_frankfurt_TXxxxx_rx4tx16
    dataSaveFilePre = dataSaveFilePrefix + "_TX"+num2str(TX_station) + "_rx" + num2str(rx_number_of_beams) + "tx" + num2str(tx_number_of_beams) + "_";
    dataTotalSaveFilePre = dataSaveFilePre + "totalRSS_best";
    dataPositionCodeFilePre = dataSaveFilePre + "posCode_best";
    dataSaveFilePre = dataSaveFilePre + "best";
    
    dataSaveFilePre = fullfile(data_save_path, dataSaveFilePre);
    dataTotalSaveFilePre = fullfile(data_save_path,dataTotalSaveFilePre);
    dataPositionCodeFilePre = fullfile(data_save_path, dataPositionCodeFilePre);

    generateRxBest = myConfig("generateRxBest")
    % whether to generate the totalRSS matrix
    generateTotal = myConfig("generateTotal")
    generatePosCodeMatrix = myConfig("generatePosCodeMatrix")

    ignoreNan = myConfig("ignoreNan");

    % dataSaveFilePre = "./dataMatrixAll_tx1_best_select1"; % only register the 1-linear index
    k = myConfig("k");
    dataSaveFilePre = dataSaveFilePre + int2str(k);
    % just as dataManipulate_all_bestk.m does   [xpos, ypos, bestTopBeamIndex1, .., bestTopBeamindexk, rss]
    dataTotalSaveFilePre = dataTotalSaveFilePre + int2str(k) ;
    dataPositionCodeFilePre = dataPositionCodeFilePre + int2str(k);
    % [ROWS, COLS]= size(bpl_table);
    %  set the tx and rx role interchanged, i.e. the tx choose a subset and rx has to sweep
    transposeTRX = myConfig("transposeTRX");
    if transposeTRX
        dataSaveFilePre = dataSaveFilePre + "_trxTranspose";
        dataTotalSaveFilePre = dataTotalSaveFilePre + "_trxTranspose";
        dataPositionCodeFilePre = dataPositionCodeFilePre + "_trxTranspose";
    end

    % if all rss data on a row is -174, remove that line
    removeDeadZone = myConfig("removeDeadZone");
    if removeDeadZone
        dataSaveFilePre = dataSaveFilePre + "_removeDead";
        dataTotalSaveFilePre = dataTotalSaveFilePre + "_removeDead";
        dataPositionCodeFilePre = dataPositionCodeFilePre + "_removeDead";
    end
    dataSaveFilePre = dataSaveFilePre + int2str(k);
    % just as dataManipulate_all_bestk.m does   [xpos, ypos, bestTopBeamIndex1, .., bestTopBeamindexk, rss]
    dataTotalSaveFilePre = dataTotalSaveFilePre + int2str(k) ;
    dataPositionCodeFilePre = dataPositionCodeFilePre + int2str(k);

    dataSaveFile = dataSaveFilePre + ".csv"
    dataTotalSaveFile = dataTotalSaveFilePre + ".csv"
    dataPosCodeSaveFile = dataPositionCodeFilePre + ".csv"


    defaultVal = -174;
    totalRSSTableSize = 16 * tx_number_of_beams * rx_number_of_beams;
    % rowIndex,colIndex,rowMax,colMax,Index,xPos,yPos
    % rowIndex, means the position where there is some non-empty data (RSS)
    % rowMax means the index corresponding to the maximum 16*64
    totalFileNum = length(saveMatFiles);
    dataMatrix = zeros(totalFileNum, (2 + k + k * 4 * tx_number_of_beams));
    dataTotalMatrix = zeros(totalFileNum, (2 + k+ totalRSSTableSize));
    dataPosCodeMatrix = zeros(totalFileNum, (2 + 2 * k));
    dataHeader = ["rowIndex", "colIndex"];
    for i = 1 : k
        dataHeader = [dataHeader, "Index" + int2str(i), "RSS" + int2str(i)];
    end
    
    if readFileName
        parpool(20);
        parfor fileIndex = 1 : length(saveMatFiles)
            % get x pos , y position
            fileName = saveMatFiles(fileIndex);
            fileNameSplit = split(fileName,".");
            fileNameSplit = fileNameSplit(1);
            fileNameSplit = split(fileNameSplit,"_");
            xpos = str2num(fileNameSplit(1));
            ypos = str2num(fileNameSplit(2));

            rssTable = getBplTableSingleEntry(save_path,xpos, ypos, meas_tuple,rxOmni, tx_number_of_beams,rx_number_of_beams,tx_phi, rx_phi);
            if sum(size(rssTable) == [4 * rx_number_of_beams, 4 * tx_number_of_beams]) ~= 2
                error("rssTable run size : " + num2str(size(rssTable)))
            end
            % save the corresponding value
            if transposeTRX
                reshape_vec = reshape(rssTable,[numel(rssTable),1]);
                error("currently do not support transposeTRX");
            else
                % make it to 64 * 16, thus each column corresponds to one measurement a rx collects at one fixed rx direction
                % reshape_vec = reshape(rssTable',[numel(rssTable),1]);
                reshape_vec = reshape(rssTable',[numel(rssTable),1]);
            end
            % [M,I] = max(rssTable,[],"all");
            % note the index  is then  [1,2,3,...N],[N+1,....]  because reshape_vec is constructed based on the transpose
            [M,I] = maxk(reshape_vec, k);
            if transposeTRX
                [colMaxs, rowMaxs] = ind2sub(size(rssTable), I);
            else
                [colMaxs, rowMaxs] = ind2sub(size(rssTable'), I);
            end
            % choose the best bp rx indices
            tempRow = [];
            for rxBestIndexIndex = 1 : length(I)
                rxBestIndex = colMaxs(rxBestIndexIndex);
                tempRow = [tempRow, rssTable(:,rxBestIndex)'];
                % convert to (row,col)
            end
            % check if only has the -174
            tempRow(isnan(tempRow)) = defaultVal;
            if removeDeadZone
                error("currently do not support remove deadzone because of parallelism (hard to maintain global variable registering which file is processed or not)");
                if sum(tempRow == defaultVal) == length(tempRow)
                    continue;
                end
            end

            % [rowMaxs, colMaxs] = ind2sub(size(rssTable), I);
            % also find the rx position
            if generateRxBest
                dataMatrixTempRow = [xpos,ypos,(I-1)', tempRow];
                % dataMatrix = [dataMatrix; dataMatrixTempRow];
                dataMatrix(fileIndex,:) = dataMatrixTempRow;
            end
            if generateTotal
                dataTotalMatrixTempRow = [xpos,ypos,(I-1)',reshape_vec'];
                % dataTotalMatrix = [dataTotalMatrix; dataTotalMatrixTempRow];
                dataTotalMatrix(fileIndex,:) = dataTotalMatrixTempRow;
            end
            if generatePosCodeMatrix
                dataPosCodeTempRow = [xpos,ypos];
                addedEntry = false;
                for i = 1 : length(rowMaxs)
                    if isnan(M(i)) && ~ignoreNan
                        error("currently do not support ignoring files because of parallelism");
                        break;
                    end
                    addedEntry = true;
                    rowMax = rowMaxs(i);
                    colMax = colMaxs(i);
                    % index subtract 1
                    dataPosCodeTempRow = [dataPosCodeTempRow, I(i)-1, M(i)];
                    % dataMatrixTempRow = [dataMatrixTempRow,  rowMax, colMax, I(i), rxPos(1:2), M(i)];
                    % dataMatrixTempRow = [dataMatrixTempRow, xpos, ypos, rowMax, colMax, I(i), rxPos(1:2)];
                end
                if addedEntry
                    dataPosCodeMatrix(fileIndex,:) = dataPosCodeTempRow;
                    % dataPosCodeMatrix = [dataPosCodeMatrix; dataPosCodeTempRow];
                end
            end



            % bpl_table{xpos,ypos} = rssTable;
            clearvars rssTable
            disp("one File processed,no " + num2str(fileIndex));
        end
    else
        error("Has to have readFileName true")
    end



    % save data
    if generateRxBest
        dataTable = array2table(dataMatrix);
        dataTable.Properties.VariableNames((end-1):end) = ["tx_pos_x", "tx_pos_y"];
        writetable(dataTable,dataSaveFile)
        disp("generate rx best at: " + dataSaveFile)
    end

    if generateTotal
        dataTotalTable = array2table(dataTotalMatrix);
        dataTotalTable.Properties.VariableNames((end-1):end) = ["tx_pos_x", "tx_pos_y"];
        writetable(dataTotalTable,dataTotalSaveFile)
        disp("generate total Matrix at: " + dataTotalSaveFile)
    end


    if generatePosCodeMatrix
        dataPosCodeTable = array2table(dataPosCodeMatrix);
        dataPosCodeTable.Properties.VariableNames(1:length(dataHeader)) = dataHeader;
        dataPosCodeTable.Properties.VariableNames((end-1):end) = ["tx_pos_x", "tx_pos_y"];

        writetable(dataPosCodeTable,dataPosCodeSaveFile)
        disp("generate pos code matrix at: " + dataPosCodeSaveFile)
    end
    delete(gcp('nocreate'));
end

function combineSingleMat(x_size,y_size,granularity, rx_number_of_beams,tx_number_of_beams,TX_station, savePathPre, matSavePathPre, chooseLOS,chooseOnlyNLOS, rxOmni)
    % savePathPre :the prefix where the input data is saved
    % matSavePathPre, the prefix folder where the combined matrix is saved

    % clear all

    meas_tuple = [];

    %% RX antenna parameters
    % 4x4 4 beams 22.5 steps
    rx_stepSize = 360 / (4 * rx_number_of_beams);
    rx_phi = -180:rx_stepSize:(180-1);

    %% TX antenna parameters
    % 8x8 16 beams 5.625 steps  16 = 64 / 4   4 quadrants
    tx_stepSize = 360 / (4 * tx_number_of_beams);
    tx_phi = -180:tx_stepSize:(180-1);

    bpl_table = cell([x_size,y_size]);

    % tx_data_file="350x350_data_OE_TX1/";

    mat_save_name = "rx_" + num2str(rx_number_of_beams) + "tx_" + num2str(tx_number_of_beams);

    % save_path= ["350x350_data_OE_TX" num2str(TX_station) "_"]
    % save_path= "350x350_data_OE_TX" + num2str(TX_station) + "_";
    % save_path= "35                                        0x350_data_OE_TX" +num2str(TX_station) + "_" + "rx" + num2str(rx_number_of_beams) + "tx" + num2str(tx_number_of_beams) + "_";

    save_path= num2str(x_size) + "x" + num2str(y_size) + "_data_OE_TX" +num2str(TX_station) + "_" + "rx" + num2str(rx_number_of_beams) + "tx" + num2str(tx_number_of_beams) + "_";
    oe_directory = save_path
    if chooseLOS
        save_path = save_path + "LOS";
        mat_save_name = mat_save_name + "_LOS";
    elseif chooseOnlyNLOS
        save_path = save_path + "onlyNLOS";
        oe_directory = oe_directory + "onlyNLOS";      
        mat_save_name = mat_save_name + "_onlyNLOS";
    end
    if rxOmni
        save_path = save_path + "rxOmni";
        % oe_directory = oe_directory + "rxOmni";
        mat_save_name = mat_save_name + "_rxOmni";
        rx_phi = 0;
    end

    save_path = fullfile(savePathPre, save_path);
    mat_save_name = mat_save_name + ".mat";
    % make dir 
    mkdir(fullfile(matSavePathPre, "TX" +num2str(TX_station)))
    mat_save_name = fullfile(matSavePathPre, "TX"+ num2str(TX_station), mat_save_name)

    % get the filePathName
    saveMatFileNames = fullfile(save_path,"matName.txt");
    if isfile(saveMatFileNames)
        saveMatFiles = readlines(saveMatFileNames,"EmptyLineRule","skip");
        readFileName = true;
    else
        readFileName = false;
    end

    if readFileName
        for fileIndex = 1 : length(saveMatFiles)
            % get x pos , y position
            fileName = saveMatFiles(fileIndex);
            fileNameSplit = split(fileName,".");
            fileNameSplit = fileNameSplit(1);
            fileNameSplit = split(fileNameSplit,"_");
            xpos = str2num(fileNameSplit(1));
            ypos = str2num(fileNameSplit(2));

            rssTable = getBplTableSingleEntry(save_path,xpos, ypos, meas_tuple,rxOmni, tx_number_of_beams,rx_number_of_beams,tx_phi, rx_phi);
            bpl_table{xpos,ypos} = rssTable;
            clearvars rssTable
        end

            % clearvars meas_tuple 
    else            % clearvars tuple_rss tuple_rxPhi tuple_rxTheta tuple_txPhi
        for x_pos = 1:granularity:x_size
            for y_pos = 1:granularity:y_size

                % % load(['bpl_data/' + save_path , num2str(x_pos), "_", num2str(y_pos),'.mat'])
                % matFileName = fullfile(save_path, num2str(x_pos)+"_"+num2str(y_pos)+'.mat');
                % if ~isfile(matFileName)
                %     % file does not exist
                %     continue;
                % end
                % load(fullfile(save_path, num2str(x_pos)+"_"+num2str(y_pos)+'.mat'));
                % if isempty(meas_tuple.rss)
                %     error("x_pos" +num2str(x_pos)+", ypos:"+num2str(y_pos) + " has empty data");
                % end
                
                % if rxOmni
                %     rssTable = -174 * ones(1, tx_number_of_beams * 4);
                % else
                %     rssTable = -174*ones(rx_number_of_beams*4, tx_number_of_beams*4);
                % end

                % for tx_beamIdLoop = 1:size(tx_phi,2)

                %     tuple_rss = meas_tuple.rss;
                %     tuple_rxPhi = meas_tuple.phi_rx;
                %     tuple_rxTheta = meas_tuple.theta_rx;
                %     tuple_txPhi = meas_tuple.phi_tx;

                %     % cut elevation to 0° AND tx_beamIdLoop
                %     log_vec = and (tuple_rxTheta == 0, tuple_txPhi == tx_phi(tx_beamIdLoop));
                %     tuple_rss = tuple_rss(log_vec);
                %     tuple_rxPhi = tuple_rxPhi(log_vec);
                %     tuple_txPhi = tuple_txPhi(log_vec);

                %     for rx_beamIdLoop = 1:size(rx_phi,2)
                %         rssTable(rx_beamIdLoop, tx_beamIdLoop) = tuple_rss(rx_beamIdLoop);
                %     end
                % end

                rssTable = getBplTableSingleEntry(save_path,x_pos, y_pos, meas_tuple,rxOmni, tx_number_of_beams,rx_number_of_beams,tx_phi, rx_phi);
                bpl_table{x_pos,y_pos} = rssTable;

                % clearvars meas_tuple 
                % clearvars tuple_rss tuple_rxPhi tuple_rxTheta tuple_txPhi
                clearvars rssTable

            end
        end
    end
        % save the variable
    save(mat_save_name, "bpl_table", "-v7.3");
    % print path
    disp("mat saved at: " + mat_save_name)
end


function [rssTable] = getBplTableSingleEntry(save_path,x_pos, y_pos, meas_tuple,rxOmni, tx_number_of_beams,rx_number_of_beams,tx_phi, rx_phi)
    % load(['bpl_data/' + save_path , num2str(x_pos), "_", num2str(y_pos),'.mat'])
    matFileName = fullfile(save_path, num2str(x_pos)+"_"+num2str(y_pos)+'.mat');
    if ~isfile(matFileName)
        % file does not exist
        rssTable = [];
        return;
    end
    load(fullfile(save_path, num2str(x_pos)+"_"+num2str(y_pos)+'.mat'));
    if isempty(meas_tuple.rss)
        error("x_pos" +num2str(x_pos)+", ypos:"+num2str(y_pos) + " has empty data");
    end
    
    if rxOmni
        rssTable = -174 * ones(1, tx_number_of_beams * 4);
    else
        rssTable = -174*ones(rx_number_of_beams*4, tx_number_of_beams*4);
    end

    tuple_rss = meas_tuple.rss;
    % tuple_rxPhi = meas_tuple.phi_rx;
    tuple_rxTheta = meas_tuple.theta_rx;
    tuple_txPhi = meas_tuple.phi_tx;

    if rxOmni
        error("currently do not support rxOmni");
    end

    if numel(tuple_rss) ~= 3 * 16 * tx_number_of_beams * rx_number_of_beams
        error("size of tuple rss mismatch!");
    end

    if numel(tuple_rxTheta) ~= 3 * 16 * tx_number_of_beams * rx_number_of_beams
        error("size of tuple rxTheta mismatch!");
    end
    
    if numel(tuple_txPhi) ~= 3 * 16 * tx_number_of_beams * rx_number_of_beams
        error("size of tuple txPhi mismatch!");
    end
    
    rssTable(1:numel(rssTable)) = tuple_rss(2:3:(end-1));
    % for tx_beamIdLoop = 1:size(tx_phi,2)


    %     % cut elevation to 0° AND tx_beamIdLoop
    %     log_vec = and (tuple_rxTheta == 0, tuple_txPhi == tx_phi(tx_beamIdLoop));
    %     tuple_rss = tuple_rss(log_vec);
    %     % tuple_rxPhi = tuple_rxPhi(log_vec);
    %     % tuple_txPhi = tuple_txPhi(log_vec);

    %     % for rx_beamIdLoop = 1:size(rx_phi,2)
    %     %     rssTable(rx_beamIdLoop, tx_beamIdLoop) = tuple_rss(rx_beamIdLoop);
    %     % end
    %     % for rx_beamIdLoop = 1:size(rx_phi,2)
    %         rssTable(:, tx_beamIdLoop) = tuple_rss(1:size(rx_phi,2));
    %     % end
    % end

    % ;

    clearvars meas_tuple 
    clearvars tuple_rss tuple_rxPhi tuple_rxTheta tuple_txPhi
    % clearvars rssTable
end

function parsave(folder_path, meas_tuple)
    save(folder_path,'meas_tuple',"-v7.3");
end
