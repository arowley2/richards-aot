%%% Generate time series of growing season rainfall parameters of day of
%%% event, depth of storm (mm), and rainfall rate (mm/hr). Generate growing
%%% season events with marked Poisson process from longterm average of data
%%% and rate of events from tipping bucket raingage.

clc;
clear all;
close all;


%%%%%Inputs %%%%%
lam=0.25;           %%% Rainfall storm arrival rate (1/day) 
al=15;              %%% Average depth of rainfall (mm)
PETm=4.5;            %%% Mean Potential ET, PET (mm/day)
PETsd=1;            %%%Std. PET (mm/day)
Tseas = 180;        %%% Growing season length (day)
Eratio=0.7;         %%%Evaporation ratio, will be negative flux at surface for BC
Nseas=1;            %%%Number of growing seasons to simulate

%%% Calculations
MAP=al*lam*Tseas;               %%%%% Total rainfall (mm)
stdp=(2*al^2*lam*Tseas)^0.5;    %%%%% Standard devation of daily rainfall (mm)
Tratio=1-Eratio;                %%% Transpiration ratio, will be sink term from root zone, uniformly taken from 0-50 cm

%%%Set up matrix storage
SummaryRainD=zeros(Tseas,Nseas);  %Storage matrix
SummaryRainR=zeros(Tseas,Nseas);  %Storage matrix

for j=1:Nseas

    %%%Generate storage precipitation events for length of growing season
    Rain=zeros(Tseas,3);

    count=0;
    while (1)
        Qsim=expinv(rand(1,1),(1/lam));     %%%%%Generate time until next storm, with exponential distribution
        Qr=ceil(Qsim);                     %%%Round up to nearest day

        %%%Update time of storm occurence
        count=count+Qr;

        %%%Loop conditions for break and update Rain with 1 for a storm event
        if count>Tseas
            break
        else
            Rain(count,1)=1;
        end
    end

    %%%Find indices of rainfall events
    Ind=find(Rain(:,1)==1);
    Tl=length(Ind);

    for i=1:Tl
        %%%%Simulate rainfall depth (mm) with exp. distribution
        Rain(Ind(i),2)=expinv(rand(1,1),al);
    end

   %%%Find indices of dry days
    Indn=find(Rain(:,1)==0);
    Tn=length(Indn);
        
    for i=1:Tn
        %%%%Simulate PET depth with norm dist with mean and std. (mm)
       Rain(Indn(i),3)=normrnd(PETm,PETsd);
    end


    clear Ind Indn Tn Tl count Qsim Qr
end


%%%%%Summary of Forcings and Top Boundary Conditions
Force(:,1)=[1:1:Tseas]';  %%%ID 1 to season length (days
Force(:,2)=Rain(:,2);     %%%% Rainfall depth (mm/day) using lambda and alpha parameters from Exp. distribution simulation
Force(:,3)=Rain(:,3);     %%%% Total ET (mm/day) using norm dist with mean and std. 
Force(:,4)=Rain(:,3)*Eratio*-1; %%%% Evaporation (mm/day), will be a upward flux at surface (opposite of rainfall)
Force(:,5)=Rain(:,3)*Tratio;    %%%% Transpiration (mm/day), will be a sink term. Take uniformly over rootzone (0-50 cm here)


%%%% Check to see if rain and ET are close. 
Totrain=sum(Rain(:,2));
TotET=sum(Rain(:,3));

writematrix(Force, 'input.txt')
