"""Descriptive v3 report: all attempted improvements, including unsuccessful ones."""

import json
import os

import numpy as np
import pandas as pd

from src.report import markdown_table
from src.data import ROOT
from src.v3.common import dense_targets, all_validation, score, load_forecast


def write_report(dataset, base, cfg, output):
    summary=pd.read_csv(output/'test-metrics.csv')
    cohorts=pd.read_csv(output/'test-cohorts.csv')
    bands=pd.read_csv(output/'test-intervals.csv')
    dense=pd.read_csv(output/'test-dense-metrics.csv')
    selection=json.loads((output/'selection-frozen.json').read_text())
    interval_selection=json.loads((output/'interval-selection-frozen.json').read_text())
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['v2_selected','selected_v3','timestral-28m_physical_672h','residual_long','residual_long_rapid','weekly_direct','weekly_residual_long']
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for name in names:
        d=summary.loc[summary.model.eq(name)]
        axes[0].plot(d.horizon_minutes,d.mae,'o-',label=name,linewidth=2 if name=='selected_v3' else 1)
    axes[0].set(xlabel='Orizzonte (minuti)',ylabel='MAE (mg/dL)',title='Confronto v2 e v3 su origini comuni',xticks=base['horizons_minutes'])
    axes[0].legend(fontsize=7,frameon=False)
    for mode,frame in bands.groupby('mode',sort=False):
        axes[1].plot(frame.horizon_minutes,frame.coverage90*100,'o-',label=mode)
    axes[1].axhline(90,color='black',linestyle=':')
    axes[1].set(xlabel='Orizzonte (minuti)',ylabel='Copertura (%)',title='Fasce nominali 90%',xticks=base['horizons_minutes'])
    axes[1].legend(fontsize=8,frameon=False)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(output/'comparison.png',dpi=180);fig.savefig(output/'comparison.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for mode,frame in bands.groupby('mode',sort=False):
        axes[0].plot(frame.horizon_minutes,frame.rapid_coverage90*100,'o-',label=mode)
        axes[1].plot(frame.horizon_minutes,frame.width90,'o-',label=mode)
    axes[0].axhline(90,color='black',linestyle=':')
    axes[0].set(title='Copertura nelle variazioni rapide',ylabel='Copertura (%)')
    axes[1].set(title='Ampiezza delle fasce',ylabel='mg/dL')
    for ax in axes:ax.set(xlabel='Minuti',xticks=base['horizons_minutes']);ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False)
    fig.savefig(output/'uncertainty.png',dpi=180);plt.close(fig)

    with np.load(output/'predictions/test_selected_v3.npz') as saved:
        positions,prediction=saved['positions'],saved['prediction']
    with np.load(output/'intervals'/f"test_{interval_selection['mode']}.npz") as saved:
        lower,upper=saved['0.9_lower'],saved['0.9_upper']
    dense_name=selection['dense_model']
    with np.load(output/'predictions'/f'test_{dense_name}.npz') as saved:
        curves=saved['curve']
    # Same fixed examples as v2; no selection based on the observed error.
    day=pd.Timestamp('2026-09-02',tz=base['timezone'])
    fig,axes=plt.subplots(1,3,figsize=(15,5.5),sharey=True)
    fig.subplots_adjust(left=.055,right=.985,top=.76,bottom=.22,wspace=.12)
    fig.suptitle('V3: previsioni e traiettoria densa',x=.055,ha='left',fontsize=20,fontweight='bold')
    fig.text(.055,.87,'2 settembre 2026 · orari prefissati · barre calibrate al 90% sui quattro orizzonti',fontsize=11,color='#555')
    for ax,hour in zip(axes,[8,12,18]):
        desired=(day+pd.Timedelta(hours=hour)).timestamp()
        row=np.searchsorted(dataset.times[positions],desired,side='right')-1
        origin=float(dataset.times[positions[row]])
        assert 0<=desired-origin<=20*60
        mask=(dataset.times>=origin-3600)&(dataset.times<=origin+10800)
        ax.axvspan(-60,0,color='#eef2f6')
        ax.plot((dataset.times[mask]-origin)/60,dataset.glucose[mask],color='#252d38',label='Glicemia osservata')
        ax.plot(np.arange(1,37)*5,curves[row],color='#e57421',linewidth=1.5,label='Modello denso separato')
        ax.errorbar(base['horizons_minutes'],prediction[row],yerr=np.stack([prediction[row]-lower[row],upper[row]-prediction[row]]),fmt='o',color='#1565c0',ecolor='#7aa7d6',capsize=4,label='Ensemble e fascia 90%')
        ax.axvline(0,color='#7b8794',linestyle='--',linewidth=1)
        at=pd.Timestamp(origin,unit='s',tz='UTC').tz_convert(base['timezone'])
        ax.set(title=f'Emessa alle {at:%H:%M}',xlabel="Minuti dall'emissione",xticks=[-60,0,30,60,120,180],xlim=(-60,190));ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('mg/dL');axes[0].legend(fontsize=7.5,frameon=False)
    fig.text(.055,.075,'La linea arancione è il modello a 36 passi scelto sulle validazioni; i quattro punti blu sono un ensemble distinto.',fontsize=10,color='#555')
    fig.text(.055,.03,'Le barre sono intervalli per singolo orizzonte. Valutazione esplorativa su un periodo già analizzato nella v2.',fontsize=10,color='#555')
    fig.savefig(output/'examples.png',dpi=160);fig.savefig(output/'examples.pdf');plt.close(fig)

    with np.load(output/'predictions'/f'test_{dense_name}.npz') as saved:
        actual=dense_targets(dataset,positions,base)
        valid=np.isfinite(actual).all(axis=1)
        excursion=np.zeros(len(actual),dtype=np.float32)
        excursion[valid]=np.ptp(actual[valid],axis=1)
        valid &= excursion>=30
        p=saved['curve'][valid];y=actual[valid]
        shape={'model':dense_name,'n_complete_windows_range_at_least_30mgdl':int(valid.sum()),
            'mean_window_max_time_error_minutes':float(np.mean(np.abs(np.argmax(p,axis=1)-np.argmax(y,axis=1))*5)),
            'mean_window_max_value_error_mgdl':float(np.mean(np.abs(p.max(axis=1)-y.max(axis=1))))}
    (output/'trajectory-shape.json').write_text(json.dumps(shape,indent=2))
    validation=[]
    for phase in ['tabular','trajectory','finetune','rolling']:
        for c in json.loads((output/f'{phase}-selection.json').read_text())['candidates']:
            for k,h in enumerate(base['horizons_minutes']):validation.append({'model':c['spec']['name'],'horizon_minutes':h,'mae':c['metrics']['mae'][k]})
    pd.DataFrame(validation).to_csv(output/'validation-metrics.csv',index=False)
    # Compare adaptation against public weights with identical context/covariates.
    valid_positions,_=all_validation(dataset,base)
    old=ROOT/cfg['v2']
    cache=old/'timestral-personal-cache'
    native_positions=np.load(cache/'positions.npy',mmap_mode='r')
    rows=np.searchsorted(native_positions,valid_positions)
    np.testing.assert_array_equal(native_positions[rows],valid_positions)
    native24=np.load(cache/'prediction.npy',mmap_mode='r')[rows]
    native72=load_forecast(old/'oof/timestral-28m_physical_72h.npz',valid_positions)
    matched=[]
    for name,pred in [('public_glucose_24h',native24),('timestral_head24',load_forecast(output/'oof/timestral_head24.npz',valid_positions)),
                      ('public_physical_72h',native72),('timestral_last72',load_forecast(output/'oof/timestral_last72.npz',valid_positions))]:
        metrics=score(dataset.targets[valid_positions],pred)
        matched.append({'model':name,**{f'MAE_{h}min':metrics['mae'][k] for k,h in enumerate(base['horizons_minutes'])}})
    pd.DataFrame(matched).to_csv(output/'finetune-matched-validation.csv',index=False)
    compact=summary.loc[summary.model.isin(['v2_selected','selected_v3','timestral-28m_physical_672h'])].pivot(index='model',columns='horizon_minutes',values='mae').reset_index()
    text='''# Esperimento personale v3\n\n## Perimetro e validità\n\nV3 implementa correttore sul Timestral 28M/covariate/28 giorni, loss pesate, traiettorie native a 36 passi, adattamento effettivo di testa/ultimo blocco Timestral, refit settimanali, ensemble senza filtro top-3 e fasce adattive. V2 resta congelato. Nessun dosaggio o comando della pompa.\n\n**Il test settembre–ottobre è già stato consultato nella v2 e ha suggerito alcune ipotesi v3. Il confronto è esplorativo retrospettivo, non una nuova conferma cieca.** Tutte le scelte v3 sono fissate sulle validazioni prima del nuovo scoring, ma la conferma richiede nuovi dati indipendenti.\n\n## Risultato sulle stesse origini\n\n'''
    text+=markdown_table(compact.rename(columns={h:f'MAE_{h}min' for h in base['horizons_minutes']}))
    selected=summary.loc[summary.model.eq('selected_v3')]
    text+='\n\nGuadagno appaiato rispetto al v2 (positivo = errore minore), con intervallo bootstrap al 95% per giornate:\n\n'
    text+=markdown_table(selected[['horizon_minutes','improvement_vs_v2','improvement_low95','improvement_high95']])
    text+='\n\nIl guadagno rispetto al v2 è contenuto, circa 2–5%, ed è maggiore sugli orizzonti lunghi. Gli intervalli bootstrap descrivono questo periodo riutilizzato e non correggono le scelte sperimentali ispirate alle analisi precedenti. La v3 include modelli con refit settimanale che possono apprendere dalle settimane già trascorse del test, mentre il v2 resta statico.\n'
    text+='\n\n![Confronto](comparison.png)\n\n## Politiche scelte prima dello scoring\n\n'
    text+=f"Ensemble: `{selection['mode']}`; fasce: `{interval_selection['mode']}`; modello denso separato: `{dense_name}`.\n\n"
    policy_groups=[(0,'tutti i regimi')] if selection['mode']=='global' else [(0,'altri periodi'),(1,'pasto recente'),(2,'variazione rapida')]
    for regime,label in policy_groups:
        for k,h in enumerate(base['horizons_minutes']):
            members=[f"{s['name']} {w:.3f}" for s,w in zip(selection['candidates'],selection['weights'][regime][k]) if w>0]
            text+=f"- {label}, {h} min: "+'; '.join(members)+'.\n'
    text+='''\n## Protocollo\n\n- Validazioni aprile/giugno/luglio, sempre con training sul passato e margine 180 minuti + 90 secondi prima del cutoff.\n- Correttori e modelli diretti usano le stesse origini ogni tre letture. Fine-tuning: origini orarie per contenere il costo, scelta delle epoche su validazione oraria e valutazione sulle origini comuni fitte.\n- Testa sola con glicemia/24 ore; testa + ultimo blocco con covariate/72 ore. Percorso differenziabile confrontato con l'inferenza nativa prima del training; pesi pubblici mai sovrascritti. I checkpoint contengono solo i parametri autorizzati.\n- Refit settimanali simulati: in ciascuna settimana il modello può usare solo esiti già osservati prima del suo aggiornamento. Nel test successivo il training può incorporare la parte precedente del test, secondo la politica predefinita. È una valutazione presequenziale, distinta dal modello statico v2.\n- Ensemble su tutti i candidati, con selezione greedy convessa e diagnostica crossfit; fallback per orizzonte al v2 se il guadagno di sviluppo non supera 0,05 mg/dL. Pesi contestuali regolarizzati verso quelli globali.\n- Scale d'errore apprese da residui fuori campione, con soli ingressi noti all'origine e ampiezza nativa Timestral. Scelta delle fasce: training scala prima di luglio, calibrazione prima del 16 luglio, confronto nella seconda metà di luglio. Le diagnostiche di sviluppo condividono la fase di selezione dei punti e non sono una stima indipendente.\n- Scala finale prima di agosto; fasce fisse calibrate su agosto. Variante rolling: errori delle ultime quattro settimane, aggiornati settimanalmente solo dopo la disponibilità del target, includendo tolleranza e orizzonte.\n- Confronti appaiati per giornate con 500 bootstrap. Stessi target osservati per tutti i modelli; nessuna esclusione basata sull'errore.\n\n## Tutti i confronti puntuali\n\n'''
    text+=markdown_table(summary[['model','horizon_minutes','n','mae','rmse','bias','improvement_vs_v2','improvement_low95','improvement_high95']])
    text+='\n\n## Errori nei regimi osservati\n\n'
    text+='Pasto recente e variazione rapida possono sovrapporsi. La rapidità usa la variazione degli ultimi 15 minuti già osservati, non la variazione futura.\n\n'
    text+=markdown_table(cohorts.loc[cohorts.model.isin(['v2_selected','selected_v3'])])
    text+='\n\n## Adattamento Timestral a parità di ingressi\n\n'
    text+='Confronto sulle medesime origini di validazione, con identico contesto e identiche covariate. Le epoche sono state scelte su queste finestre: è una diagnostica di sviluppo. Il benchmark Timestral di 28 giorni resta separato.\n\n'
    text+=markdown_table(pd.DataFrame(matched))
    text+='\n\n## Fasce: copertura, variazioni rapide e ampiezza\n\n'
    text+=markdown_table(bands)
    text+='\n\n![Incertezza](uncertainty.png)\n\nLe fasce sono per il singolo orizzonte, non simultanee sull’intera curva. Copertura e ampiezza vanno lette insieme; il periodo contiene poche variazioni rapide e osservazioni correlate.\n'
    text+='\nLe future azioni di Control-IQ, i pasti non ancora registrati e le somministrazioni future non vengono forniti al modello. Restano fonti di errore, soprattutto sugli orizzonti lunghi. Le conclusioni riguardano il dominio glicemico effettivamente osservato (minimo 75 mg/dL) e lo storico disponibile della pompa.\n'
    text+='\n## Traiettorie dense\n\n'
    text+=markdown_table(dense.groupby('model',sort=False).mae.mean().reset_index().rename(columns={'mae':'MAE_medio_36_passi'}))
    text+=f"\n\nSul modello denso scelto: {shape['n_complete_windows_range_at_least_30mgdl']} finestre complete con escursione osservata almeno 30 mg/dL; errore medio del tempo del massimo nella finestra {shape['mean_window_max_time_error_minutes']:.1f} minuti, errore dell'ampiezza del massimo {shape['mean_window_max_value_error_mgdl']:.1f} mg/dL. Il massimo della finestra può cadere al suo bordo e non coincide necessariamente con un picco fisiologico completo.\n"
    text+='\n![Esempi prefissati](examples.png)\n\n## Artefatti e ripresa\n\n`selection-frozen.json`, `interval-selection-frozen.json`, manifest e `selection-before-scoring.json` conservano le scelte. `models/`, `weekly-models/` e `finetune-models/` contengono checkpoint riutilizzabili. `test-metrics.csv`, `test-cohorts.csv`, `test-intervals.csv` e `test-dense-metrics.csv` espongono gli esiti, anche dei tentativi senza beneficio. `verified.json` viene scritto solo dopo i controlli finali.\n'
    (output/'report.md').write_text(text)
