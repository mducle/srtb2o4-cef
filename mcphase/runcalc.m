swo = spinw('SrTb2O4_30341-ICSD.cif')
set_rkky(0.4, 0.5);
powspec = swo.powspec(linspace(0,3,15), 'Evect', linspace(0,40,800), 'nRand', 500, 'fibo', true)
pw2 = sw_instrument(powspec, 'dE', eres, 'dQ', 0.01, 'Ei', 7, 'thetaMin', 5, 'thetaMax', 135);
en = (pw2.Evect(1:end-1) + pw2.Evect(2:end))/2;
figure;
subplot(121); sw_plotspec(pw2); caxis([0 1]); ylim([0 3.5]); xlim([0 3.5]);
subplot(122); plot(nansum(real(pw2.swConv(:,1:100)),2), en, '.-'); ylim([0 3.5])

%{
% Runs a loop over k-values, scaling A values to keep constant bandwidth
aval = @(k) (0.4/(3.45^2)).*((2.*k.*3.45).^2)
kval=0.3:0.01:1
for kk=1:numel(kval);
    set_rkky(aval(kval(kk)),kval(kk));
    powspec_k(kk) = swo.powspec(linspace(0,3,15), 'Evect', linspace(0,40,800), 'nRand', 30, 'fibo', true);
end
figure; set(gcf, 'Position', [680 433 1234 420]);
for kk=1:numel(powspec_k);
    pw2 = sw_instrument(powspec_k(kk), 'dE', eres, 'dQ', 0.01, 'Ei', 7, 'thetaMin', 5, 'thetaMax', 135);
    clf; subplot(121); sw_plotspec(pw2); caxis([0 1]); ylim([0 3.5]); xlim([0 3.5]);
    en = (pw2.Evect(1:end-1) + pw2.Evect(2:end))/2;
    subplot(122); plot(nansum(real(pw2.swConv(:,1:min([100 size(pw2.swConv,2)]))),2), en, '.-'); ylim([0 3.5]);
    pause(1);
end
%}

set_exchange(@(r)-0.38./(r.^3))
powspec = swo.powspec(linspace(0,3,150), 'Evect', linspace(0,40,800), 'nRand', 500, 'fibo', true)
pw2 = sw_instrument(powspec, 'dE', eres, 'dQ', 0.01, 'Ei', 7, 'thetaMin', 5, 'thetaMax', 135);
en = (pw2.Evect(1:end-1) + pw2.Evect(2:end))/2;
figure;
subplot(121); sw_plotspec(pw2); caxis([0 1]); ylim([0 3.5]); xlim([0 3.5]);
subplot(122); plot(nansum(real(pw2.swConv(:,1:100)),2), en, '.-'); ylim([0 3.5])
