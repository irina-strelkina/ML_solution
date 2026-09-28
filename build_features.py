from __future__ import annotations
import numpy as np
import pandas as pd

EVENT_TYPES = ['item_view','search_results_view','photo_swipe','favorite_add','seller_page_view','contact_phone_show','captcha_shown','login','contact_chat_open','contact_message_sent']
PLATFORMS = ['web','desktop','android','ios']
UA_FAMILIES = ['headless_chrome','scrapy','python_requests','python_urllib','curl','node_fetch','go_http','okhttp','yandex_browser','firefox','chrome_mobile','chrome','safari_mobile','safari','other']
UA_OSES = ['android','ios','windows','mac','linux','other']


def normalize_platform(s):
    return s.astype('string').str.strip().str.lower().replace({'iphone':'ios'})

def ua_family(ua):
    s=(ua or '').lower()
    if 'headlesschrome' in s:return 'headless_chrome'
    if 'scrapy' in s:return 'scrapy'
    if 'python-requests' in s:return 'python_requests'
    if 'python-urllib' in s:return 'python_urllib'
    if s.startswith('curl/') or ' curl/' in s:return 'curl'
    if 'node-fetch' in s:return 'node_fetch'
    if 'go-http-client' in s:return 'go_http'
    if 'okhttp' in s:return 'okhttp'
    if 'yabrowser' in s:return 'yandex_browser'
    if 'firefox' in s:return 'firefox'
    if 'chrome' in s and 'mobile' in s:return 'chrome_mobile'
    if 'chrome' in s:return 'chrome'
    if 'safari' in s and ('iphone' in s or 'mobile' in s):return 'safari_mobile'
    if 'safari' in s:return 'safari'
    return 'other'

def ua_os(ua):
    s=(ua or '').lower()
    if 'android' in s:return 'android'
    if 'iphone' in s or 'ios' in s:return 'ios'
    if 'windows' in s:return 'windows'
    if 'macintosh' in s or 'mac os' in s:return 'mac'
    if 'linux' in s or 'x11' in s:return 'linux'
    return 'other'

def filter_events(events, meta):
    ev=events.merge(meta[['cookie_id','window_start_ts','window_end_ts']],on='cookie_id',how='inner')
    return ev[(ev.event_ts>=ev.window_start_ts)&(ev.event_ts<ev.window_end_ts)].copy()

def add_count_frac(base, ev, col, prefix, categories):
    ct=pd.crosstab(ev.cookie_id, ev[col]).reindex(columns=categories, fill_value=0)
    ct=ct.reindex(base.index, fill_value=0)
    n=base['n_events'].replace(0,np.nan)
    for c in categories:
        safe=str(c).replace(' ','_')
        base[f'cnt_{prefix}__{safe}']=ct[c].astype(float)
        base[f'frac_{prefix}__{safe}']=(ct[c]/n).fillna(0).astype(float)
        base[f'has_{prefix}__{safe}']=(ct[c]>0).astype(int)
    arr=ct.to_numpy(float)
    s=arr.sum(axis=1,keepdims=True)
    p=np.divide(arr,s,out=np.zeros_like(arr),where=s>0)
    base[f'{prefix}_entropy']=np.where(p>0,-p*np.log(p+1e-15),0).sum(axis=1)
    base[f'{prefix}_max_share']=np.divide(arr.max(axis=1),s[:,0],out=np.zeros(len(arr)),where=s[:,0]>0)

def cat_distribution_stats(ev, col, prefix, base):
    z=ev.loc[ev[col].notna(),['cookie_id',col]].groupby(['cookie_id',col],observed=True).size().rename('cnt').reset_index()
    if z.empty:
        base[f'{prefix}_entropy']=0.;base[f'{prefix}_max_share']=0.;return
    totals=z.groupby('cookie_id').cnt.transform('sum')
    z['p']=z.cnt/totals
    ent=-(z.p*np.log(z.p)).groupby(z.cookie_id).sum()
    mx=z.groupby('cookie_id').p.max()
    base[f'{prefix}_entropy']=ent.reindex(base.index).fillna(0)
    base[f'{prefix}_max_share']=mx.reindex(base.index).fillna(0)

def mode_series(ev,col):
    z=ev.loc[ev[col].notna(),['cookie_id',col]].groupby(['cookie_id',col],observed=True).size().rename('cnt').reset_index()
    if z.empty:return pd.Series(dtype=object)
    z=z.sort_values(['cookie_id','cnt',col],ascending=[True,False,True]).drop_duplicates('cookie_id')
    return z.set_index('cookie_id')[col].astype(str)

def build_features(events,meta):
    ev=filter_events(events,meta)
    ev['platform_norm']=normalize_platform(ev.platform)
    ev['ua_family']=ev.user_agent.fillna('').map(ua_family)
    ev['ua_os']=ev.user_agent.fillna('').map(ua_os)
    ev['rel_sec']=(ev.event_ts-ev.window_start_ts).dt.total_seconds()
    ev['event_hour']=ev.event_ts.dt.hour
    ev=ev.sort_values(['cookie_id','event_ts'],kind='mergesort')

    g=ev.groupby('cookie_id',sort=False,observed=True)
    base=pd.DataFrame(index=pd.Index(meta.cookie_id,name='cookie_id'))
    base['n_events']=g.size().reindex(base.index).fillna(0).astype(int)
    # nunique/missingness/counts
    cols=[('eid','eid'),('event_name','event'),('platform','platform_raw'),('platform_norm','platform'),('user_agent','ua'),('ua_family','ua_family'),('ua_os','ua_os'),('item_id','item'),('item_category','category'),('item_location','location'),('seller_type','seller'),('search_query','query'),('search_page','search_page'),('pointer_x','pointer_x'),('pointer_y','pointer_y')]
    for c,p in cols:
        base[f'{p}_nunique']=g[c].nunique(dropna=True).reindex(base.index).fillna(0)
        cnt=g[c].count().reindex(base.index).fillna(0)
        base[f'{p}_missing_frac']=1-cnt/base.n_events.replace(0,np.nan)
        base[f'{p}_missing_frac']=base[f'{p}_missing_frac'].fillna(1)

    add_count_frac(base,ev,'event_name','event',EVENT_TYPES)
    add_count_frac(base,ev,'platform_norm','platform',PLATFORMS)
    add_count_frac(base,ev,'ua_family','uafam',UA_FAMILIES)
    add_count_frac(base,ev,'ua_os','uaos',UA_OSES)

    # categorical concentration
    for c,p in [('item_id','item'),('item_category','category'),('item_location','location'),('seller_type','seller'),('search_query','query')]:
        cat_distribution_stats(ev,c,p,base)
        nonnull=g[c].count().reindex(base.index).fillna(0)
        base[f'{p}_unique_ratio']=base[f'{p}_nunique']/nonnull.replace(0,np.nan)
        base[f'{p}_unique_ratio']=base[f'{p}_unique_ratio'].fillna(0)

    # categorical modes for CatBoost
    for c,p in [('user_agent','ua_mode'),('ua_family','ua_family_mode'),('ua_os','ua_os_mode'),('platform_norm','platform_mode'),('seller_type','seller_mode'),('item_category','category_mode'),('item_location','location_mode'),('search_query','query_mode'),('platform','platform_raw_mode')]:
        base[p]=mode_series(ev,c).reindex(base.index).fillna('__MISSING__').astype(str)

    # search page numeric
    ev['search_page_num']=pd.to_numeric(ev.search_page,errors='coerce')
    spg=ev.groupby('cookie_id').search_page_num
    for stat in ['mean','std','min','max','median']:
        s=getattr(spg,stat)().reindex(base.index).fillna(0)
        base[f'search_page_{stat}']=s
    base['search_page_p90']=spg.quantile(.9).reindex(base.index).fillna(0)
    for th in [1,3,5,10,20]:
        s=(ev.search_page_num>th).groupby(ev.cookie_id).sum().reindex(base.index).fillna(0)
        denom=spg.count().reindex(base.index).fillna(0).replace(0,np.nan)
        base[f'search_page_gt{th}_frac']=(s/denom).fillna(0)

    # time offsets + gaps
    base['first_event_offset_sec']=g.rel_sec.first().reindex(base.index).fillna(0)
    base['last_event_offset_sec']=g.rel_sec.last().reindex(base.index).fillna(0)
    base['activity_span_sec']=base.last_event_offset_sec-base.first_event_offset_sec
    base['activity_span_frac_day']=base.activity_span_sec/86400
    base['events_per_active_hour']=base.n_events/(base.activity_span_sec/3600).clip(lower=1/3600)
    ev['gap_sec']=g.rel_sec.diff()
    gg=ev.groupby('cookie_id').gap_sec
    for stat in ['mean','std','min','max','median']:
        base[f'gap_{stat}_sec']=getattr(gg,stat)().reindex(base.index).fillna(0)
    for q in [.1,.25,.75,.9]:base[f'gap_p{int(q*100)}_sec']=gg.quantile(q).reindex(base.index).fillna(0)
    base['gap_cv']=base.gap_std_sec/(base.gap_mean_sec+1e-6)
    gapden=gg.count().reindex(base.index).fillna(0).replace(0,np.nan)
    for th in [0,1,2,5,10,30,60,300,600,1800,3600]:
        num=(ev.gap_sec<=th).groupby(ev.cookie_id).sum().reindex(base.index).fillna(0)
        base[f'gap_le_{th}_frac']=(num/gapden).fillna(0)

    # burst buckets
    for secs,pref in [(1,'sec'),(60,'min'),(300,'5min'),(3600,'hour')]:
        ev[f'bucket_{pref}']=(ev.rel_sec//secs).astype('int32')
        z=ev.groupby(['cookie_id',f'bucket_{pref}'],observed=True).size().rename('cnt')
        by=z.groupby(level=0)
        base[f'active_{pref}_buckets']=by.size().reindex(base.index).fillna(0)
        base[f'max_events_per_{pref}']=by.max().reindex(base.index).fillna(0)
        base[f'mean_events_per_active_{pref}']=by.mean().reindex(base.index).fillna(0)
    base['active_hour_of_day_nunique']=g.event_hour.nunique().reindex(base.index).fillna(0)
    # hour entropy and time-of-day fractions
    cat_distribution_stats(ev,'event_hour','hour',base)
    base['night_event_frac']=(((ev.event_hour<7)|(ev.event_hour>=23)).groupby(ev.cookie_id).mean().reindex(base.index).fillna(0))
    base['workhour_event_frac']=(((ev.event_hour>=9)&(ev.event_hour<19)).groupby(ev.cookie_id).mean().reindex(base.index).fillna(0))

    # transition features vectorized
    ev['prev_event']=g.event_name.shift(1)
    valid=ev.prev_event.notna()
    transden=valid.groupby(ev.cookie_id).sum().reindex(base.index).fillna(0).replace(0,np.nan)
    base['same_event_consecutive_frac']=((ev.event_name==ev.prev_event)&valid).groupby(ev.cookie_id).sum().reindex(base.index).fillna(0)/transden
    base['same_event_consecutive_frac']=base.same_event_consecutive_frac.fillna(0)
    pairs=[('search_results_view','item_view'),('item_view','item_view'),('item_view','photo_swipe'),('item_view','seller_page_view'),('item_view','contact_phone_show'),('item_view','favorite_add'),('search_results_view','search_results_view'),('seller_page_view','item_view'),('captcha_shown','search_results_view'),('captcha_shown','item_view')]
    for a,b in pairs:
        num=((ev.prev_event==a)&(ev.event_name==b)).groupby(ev.cookie_id).sum().reindex(base.index).fillna(0)
        base[f'trans__{a}__{b}']=(num/transden).fillna(0)
    ev['trans_key']=ev.prev_event.fillna('START')+'>'+ev.event_name.astype(str)
    base['transition_nunique']=ev.loc[valid].groupby('cookie_id').trans_key.nunique().reindex(base.index).fillna(0)
    tz=ev.loc[valid].groupby(['cookie_id','trans_key'],observed=True).size().rename('cnt')
    if len(tz):
        mx=tz.groupby(level=0).max().reindex(base.index).fillna(0)
        base['transition_max_share']=(mx/transden).fillna(0)
    else:base['transition_max_share']=0

    # search page sequential diffs
    se=ev.loc[ev.search_page_num.notna(),['cookie_id','event_ts','search_page_num']].copy().sort_values(['cookie_id','event_ts'])
    se['spdiff']=se.groupby('cookie_id').search_page_num.diff()
    sd=se.groupby('cookie_id').spdiff
    base['search_page_diff_mean']=sd.mean().reindex(base.index).fillna(0)
    base['search_page_diff_abs_mean']=se.spdiff.abs().groupby(se.cookie_id).mean().reindex(base.index).fillna(0)
    sden=sd.count().reindex(base.index).fillna(0).replace(0,np.nan)
    for name,mask in [('inc1',se.spdiff==1),('nondec',se.spdiff>=0),('jump_gt3',se.spdiff.abs()>3)]:
        base[f'search_page_{name}_frac']=(mask.groupby(se.cookie_id).sum().reindex(base.index).fillna(0)/sden).fillna(0)

    # pointer stats
    ev['px']=pd.to_numeric(ev.pointer_x,errors='coerce');ev['py']=pd.to_numeric(ev.pointer_y,errors='coerce')
    pm=ev.px.notna()&ev.py.notna()
    base['pointer_present_count']=pm.groupby(ev.cookie_id).sum().reindex(base.index).fillna(0)
    base['pointer_present_frac']=base.pointer_present_count/base.n_events.replace(0,np.nan);base.pointer_present_frac=base.pointer_present_frac.fillna(0)
    for c in ['px','py']:
        pg=ev.loc[pm].groupby('cookie_id')[c]
        base[f'pointer_{c[-1]}_std']=pg.std().reindex(base.index).fillna(0)
        base[f'pointer_{c[-1]}_mean']=pg.mean().reindex(base.index).fillna(0)
        base[f'pointer_{c[-1]}_min']=pg.min().reindex(base.index).fillna(0)
        base[f'pointer_{c[-1]}_max']=pg.max().reindex(base.index).fillna(0)
        base[f'pointer_{c[-1]}_range']=base[f'pointer_{c[-1]}_max']-base[f'pointer_{c[-1]}_min']
        base[f'pointer_{c[-1]}_p10']=pg.quantile(.1).reindex(base.index).fillna(0)
        base[f'pointer_{c[-1]}_p90']=pg.quantile(.9).reindex(base.index).fillna(0)
    # fixed edges / suspicious coordinate signatures
    base['pointer_x_zero_frac']=(ev.loc[pm].px.eq(0).groupby(ev.loc[pm].cookie_id).mean().reindex(base.index).fillna(0))
    base['pointer_y_zero_frac']=(ev.loc[pm].py.eq(0).groupby(ev.loc[pm].cookie_id).mean().reindex(base.index).fillna(0))
    base['pointer_y_1080_frac']=(ev.loc[pm].py.eq(1080).groupby(ev.loc[pm].cookie_id).mean().reindex(base.index).fillna(0))
    base['pointer_x_ge1600_frac']=(ev.loc[pm].px.ge(1600).groupby(ev.loc[pm].cookie_id).mean().reindex(base.index).fillna(0))
    base['pointer_x_le1200_frac']=(ev.loc[pm].px.le(1200).groupby(ev.loc[pm].cookie_id).mean().reindex(base.index).fillna(0))
    ev['coord_key']=ev.px.astype('string')+'_'+ev.py.astype('string')
    base['pointer_coord_nunique']=ev.loc[pm].groupby('cookie_id').coord_key.nunique().reindex(base.index).fillna(0)
    base['pointer_coord_unique_ratio']=base.pointer_coord_nunique/base.pointer_present_count.replace(0,np.nan);base.pointer_coord_unique_ratio=base.pointer_coord_unique_ratio.fillna(0)
    ev['prev_px']=g.px.shift(1);ev['prev_py']=g.py.shift(1)
    mv=pm & ev.prev_px.notna() & ev.prev_py.notna()
    ev['pointer_move']=np.hypot(ev.px-ev.prev_px,ev.py-ev.prev_py)
    base['pointer_move_mean']=ev.loc[mv].groupby('cookie_id').pointer_move.mean().reindex(base.index).fillna(0)
    base['pointer_move_median']=ev.loc[mv].groupby('cookie_id').pointer_move.median().reindex(base.index).fillna(0)
    base['pointer_move_zero_frac']=(ev.loc[mv].pointer_move.eq(0).groupby(ev.loc[mv].cookie_id).mean().reindex(base.index).fillna(0))

    # duplicates/same timestamp
    raw_cols=['cookie_id','event_ts','eid','event_name','platform','user_agent','item_id','item_category','item_location','seller_type','search_query','search_page','pointer_x','pointer_y']
    ev['is_exact_dup']=ev.duplicated(subset=raw_cols,keep=False)
    ev['is_key_dup']=ev.duplicated(subset=['cookie_id','event_ts','eid'],keep=False)
    for c,p in [('is_exact_dup','exact_dup'),('is_key_dup','key_dup')]:
        base[f'{p}_count']=ev[c].groupby(ev.cookie_id).sum().reindex(base.index).fillna(0)
        base[f'{p}_frac']=ev[c].groupby(ev.cookie_id).mean().reindex(base.index).fillna(0)
    ev['same_ts']=ev.duplicated(subset=['cookie_id','event_ts'],keep=False)
    base['same_timestamp_frac']=ev.same_ts.groupby(ev.cookie_id).mean().reindex(base.index).fillna(0)

    # hand-crafted behavioral interaction ratios
    eps=1e-6
    base['locations_per_category']=base.location_nunique/(base.category_nunique+eps)
    base['items_per_category']=base.item_nunique/(base.category_nunique+eps)
    base['locations_per_item']=base.location_nunique/(base.item_nunique+eps)
    base['items_per_location']=base.item_nunique/(base.location_nunique+eps)
    base['max_page_per_query']=base.search_page_max/(base.query_nunique+1)
    base['p90_page_per_query']=base.search_page_p90/(base.query_nunique+1)
    base['events_per_item']=base.n_events/(base.item_nunique+1)
    base['events_per_location']=base.n_events/(base.location_nunique+1)
    base['search_to_item_ratio']=base['cnt_event__search_results_view']/(base['cnt_event__item_view']+1)
    base['item_to_search_ratio']=base['cnt_event__item_view']/(base['cnt_event__search_results_view']+1)
    base['human_action_frac']=(base['cnt_event__photo_swipe']+base['cnt_event__favorite_add']+base['cnt_event__contact_phone_show']+base['cnt_event__contact_chat_open']+base['cnt_event__contact_message_sent']+base['cnt_event__login'])/base.n_events.clip(lower=1)
    base['scrape_action_frac']=(base['cnt_event__search_results_view']+base['cnt_event__item_view']+base['cnt_event__seller_page_view'])/base.n_events.clip(lower=1)
    base['contact_action_frac']=(base['cnt_event__contact_phone_show']+base['cnt_event__contact_chat_open']+base['cnt_event__contact_message_sent'])/base.n_events.clip(lower=1)
    base['breadth_depth_score']=base.location_nunique*np.log1p(base.search_page_max)/(base.category_nunique+1)
    base['speed_breadth_score']=base.mean_events_per_active_min*np.log1p(base.location_nunique)
    base['fast_gap_breadth_score']=base.gap_le_60_frac*np.log1p(base.location_nunique)*np.log1p(base.n_events)

    # metadata
    m=meta.set_index('cookie_id').reindex(base.index)
    age=(m.window_start_ts-m.cookie_created_at).dt.total_seconds().clip(lower=0)
    base['cookie_age_sec']=age
    base['cookie_age_hours']=age/3600
    base['cookie_age_days']=age/86400
    base['log1p_cookie_age_hours']=np.log1p(base.cookie_age_hours)
    base['created_hour']=m.cookie_created_at.dt.hour
    base['created_dow']=m.cookie_created_at.dt.dayofweek
    for h in [1,6,24,72,168,720]:
        base[f'cookie_age_le_{h}h']=base.cookie_age_hours.le(h).astype(int)
    # browser/app major version extracted from modal user-agent
    ua=base['ua_mode'].astype(str)
    import re
    def _major(v):
        pats=[r'HeadlessChrome/(\d+)',r'Chrome/(\d+)',r'Firefox/(\d+)',r'Avito/(\d+)']
        for pat in pats:
            m=re.search(pat,v)
            if m:return float(m.group(1))
        return 0.0
    base['ua_major_version']=ua.map(_major)

    base=base.replace([np.inf,-np.inf],np.nan)
    cat_cols=['ua_mode','ua_family_mode','ua_os_mode','platform_mode','seller_mode','category_mode','location_mode','query_mode','platform_raw_mode']
    for c in cat_cols:base[c]=base[c].fillna('__MISSING__').astype(str)
    for c in base.columns:
        if c not in cat_cols:base[c]=pd.to_numeric(base[c],errors='coerce').fillna(0)
    return base.reset_index()

if __name__=='__main__':
    train=pd.read_csv('data/train.csv',parse_dates=['cookie_created_at','window_start_ts','window_end_ts'])
    test=pd.read_csv('data/test.csv',parse_dates=['cookie_created_at','window_start_ts','window_end_ts'])
    events=pd.read_csv('data/events.csv.gz',parse_dates=['event_ts'])
    xtr=build_features(events,train);xte=build_features(events,test)
    xtr.to_pickle('features_train.pkl');xte.to_pickle('features_test.pkl')
    print('features',xtr.shape,xte.shape)
