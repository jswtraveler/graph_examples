import pyplot.matplotlib as py
import seaborn as sns

#Cumulative vs Dist Bar Plot

fig, ax = plt.subplots(figsize=(7,7))
sns.histplot(data=data, x="service_level", bins=30,cumulative=True, ax=ax, color='red', label='cumulative')
#plt.hist(data['service_level'], density=True, bins=30, color='red')
ax2=ax.twinx()
ax.set(ylabel='cumulative count')
sns.histplot(data=data, x="service_level", bins=30,cumulative=False, ax=ax2, label='distribution')
ax2.set(ylabel='count')
ax.legend(loc='upper right')
ax2.legend(loc='upper left')

#Box & whisker shortage plot

df_current = shortage_overall[['dt','fleet','days_prior','seat','current_shortage_error'
                              ]].copy().rename(columns={'current_shortage_error':'shortage'})
df_current['method'] = 'current'
df_m4 = shortage_overall[['dt','fleet','days_prior','seat','m4_shortage_error'
                         ]].copy().rename(columns={'m4_shortage_error':'shortage'})
df_m4['method'] = 'm4'
shortage_box = pd.concat([df_current, df_m4])

fig, ax = plt.subplots(figsize=(20,10))
fig.suptitle('2020-2021')
sns.boxplot(x='fleet', y='shortage', hue = 'method', data=shortage_box, ax=ax, showfliers=False)
plt.axhline(y=0, color='red')
plt.ylabel('shortage error')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(20,10))
sns.boxplot(x='fleet', y='shortage', hue = 'method', data=shortage_box[shortage_box.dt>=pd.to_datetime('2021-01-01')], ax=ax, showfliers=False)
fig.suptitle('2021')
plt.ylabel('shortage error')
plt.axhline(y=0, color='red')
plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(20,10))
sns.boxplot(x='fleet', y='shortage', hue = 'method'
, data=shortage_box[(shortage_box.dt>=pd.to_datetime('2020-11-24')) &
                   (shortage_box.dt<=pd.to_datetime('2020-11-30'))
                   ], ax=ax, showfliers=False)
fig.suptitle('Thanksgiving 2020')
plt.ylabel('shortage error')
plt.axhline(y=0, color='red')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(20,10))
sns.boxplot(x='fleet', y='shortage', hue = 'method'
, data=shortage_box[
    ((shortage_box.dt>=pd.to_datetime('2020-11-24')) &
   (shortage_box.dt<=pd.to_datetime('2020-11-30')))
    |
    ((shortage_box.dt>=pd.to_datetime('2020-12-23')) &
   (shortage_box.dt<=pd.to_datetime('2020-12-26')))
    |
    ((shortage_box.dt>=pd.to_datetime('2021-04-01')) &
   (shortage_box.dt<=pd.to_datetime('2021-04-05')))
    |
    ((shortage_box.dt == pd.to_datetime('2021-04-10')))
   ], ax=ax, showfliers=False)
fig.suptitle('Holidays')
plt.ylabel('shortage error')
plt.axhline(y=0, color='red')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(20,10))
sns.boxplot(x='fleet', y='shortage', hue = 'method'
, data=shortage_box[
    ((shortage_box.dt>=pd.to_datetime('2020-12-01')) &
   (shortage_box.dt<=pd.to_datetime('2020-12-10')))
    |
    ((shortage_box.dt>=pd.to_datetime('2021-04-06')) &
   (shortage_box.dt<=pd.to_datetime('2021-04-14')))

   ], ax=ax, showfliers=False)
plt.axhline(y=0, color='red')
fig.suptitle('After Holidays')
plt.ylabel('shortage error')
plt.show()
plt.close()

#############
##Accuracy function
#############
#To add columns to compare against current logic add the column name to a list in the columns arg

# To add custom date filters to the box and whisker add them in the following format:
# ["dt>= '2021-01-01' & ...",]
#If you wnat multiple dates in the same graph add them within a single double quotation
#If you want to look at different dates separately add them as separate statements
#in a list


def shortage_accuracy(sql_change=None, columns=None, date_query = None, db, user, pw):
    import pandas as pd
    import numpy as np
    import teradatasql
    import seaborn as sns
    import matplotlib.pyplot as plt
    import datetime
    import warnings

    cnxn= teradatasql.connect(None, host = db, user = user, password = pw, LOGMECH = 'TD2') 
    
    q = """
    select * from zhtp_oap.PilotShortage_vw
    where DT>= date '2020-01-01'
    and FLEET in ('7ER','73N','320')
    """
    if sql_change:
        q = q + sql_change
    
    print('Pulling Data from zhtp.oap.PilotShortage_vw')
    shortage = pd.read_sql(q, cnxn)
    shortage.columns = [a.lower() for a in shortage.columns]
    shortage.dt = pd.to_datetime(shortage.dt)
    shortage['rsrc_perc_incr'] = shortage['rsrc_perc_3dp'] - shortage['rsrc_perc']
    shortage['open_perc_incr'] = shortage['open_perc_3dp'] - shortage['open_perc']
    
    shortage['supply_m_cnt'] = np.round(np.where(shortage.days_prior>=7,
    (shortage['m3_supply_pred']),
    (shortage['m2_supply_pred'])),0)

    shortage['demand_m_cnt'] = np.round(np.where(shortage.days_prior>=7,
    (shortage['m3_demand_pred']),
    (shortage['m2_demand_pred'])),0)

    #m4
    
    #shortage.rename(columns={'m4_supply_cnt':'supply_m4_cnt'},inplace=True)
    
    shortage['3dp_shortage'] = shortage.open_duty_ct_3dp - shortage.resource_count_3dp

    shortage['current_supply_err'] = shortage['supply_m_cnt'] - shortage['resource_count_3dp']
    shortage['current_supply_err_rt'] = shortage['current_supply_err']/shortage['resource_count_3dp']
    shortage['abs_current_supply_err_rt'] = np.abs(shortage['current_supply_err_rt'])
    
    shortage['current_demand_err'] = shortage['demand_m_cnt'] - shortage['open_duty_ct_3dp']
    shortage['current_demand_err_rt'] = shortage['current_demand_err']/shortage['open_duty_ct_3dp']
    shortage['abs_current_demand_err_rt'] = np.abs(shortage['current_demand_err_rt'])
    
    shortage['current_shortage'] = shortage.demand_m_cnt - shortage.supply_m_cnt

    shortage['current_shortage_error'] = shortage['3dp_shortage'] - shortage['current_shortage']

    abs_supply_err_col = ['abs_current_supply_err_rt']
    supply_err_col = ['current_supply_err_rt']

    
    if columns:
        print('adding columns')
        for col in columns:
            print(col)
            shortage[col + '_err'] = shortage[col] - shortage['resource_count_3dp']
            shortage[col + '_supply_err_rt'] = shortage[col + '_err']/shortage['resource_count_3dp']
            shortage['abs_' + col + '_supply_err_rt'] = np.abs(shortage[col + '_supply_err_rt'])
            shortage[col + '_shortage'] = shortage.demand_m_cnt - shortage[col]
            abs_supply_err_col = abs_supply_err_col + ['abs_' + col + '_supply_err_rt']
            supply_err_col = supply_err_col + [col + '_supply_err_rt']

    print('\n     Last 14 Days Supply Absolute Error Rate Mean:')
    print((shortage[shortage.dt>=(datetime.datetime.now() - datetime.timedelta(days=15))]
        .dropna(subset=['resource_count_3dp']).sort_values(['fleet','seat'])
        .groupby(['fleet','seat'])[abs_supply_err_col].mean().round(3)))

    print('\n     2021 Supply Error Rate Mean:')
    print((shortage[shortage.dt>=(datetime.datetime.now() - datetime.timedelta(days=15))]
        .dropna(subset=['resource_count_3dp']).sort_values(['fleet','seat'])
        .groupby(['fleet','seat'])[supply_err_col].mean().round(3)))

    print('\n     2021 Supply Absolute Error Rate Mean:')
    print((shortage[shortage.dt>=pd.to_datetime('2021-01-01')].dropna(subset=['resource_count_3dp'])
        .sort_values(['fleet','seat']).groupby(['fleet','seat'])[abs_supply_err_col].mean().round(3)))

    print('\n     2021 Supply Error Rate Mean:')
    print((shortage[shortage.dt>=pd.to_datetime('2021-01-01')].dropna(subset=['resource_count_3dp'])
        .sort_values(['fleet','seat']).groupby(['fleet','seat'])[supply_err_col].mean().round(3)))

    print('\n     2020-2021 Supply Absolute Error Rate Mean:')
    print((shortage.dropna(subset=['resource_count_3dp']).sort_values(['fleet','seat'])
        .groupby(['fleet','seat'])[abs_supply_err_col].mean().round(3)))

    print('\n     2020-2021 Supply Error Rate Mean:')
    print((shortage.dropna(subset=['resource_count_3dp']).sort_values(['fleet','seat'])
        .groupby(['fleet','seat'])[supply_err_col].mean().round(3)))
    
    #Supply graphs
    #Box & whisker supply plot
    df_current = shortage[['dt','fleet','days_prior','seat','current_supply_err_rt'
        ,'abs_current_supply_err_rt']].copy().rename(columns={'current_supply_err_rt':'error_rate'
        ,'abs_current_supply_err_rt':'abs_error_rate'})
    df_current['method'] = 'current'

    shortage_box = df_current.copy()

    if columns:
        for col in columns:
            df_new = shortage[['dt','fleet','days_prior','seat',col + '_supply_err_rt'
                ,'abs_'+ col + '_supply_err_rt']].copy().rename(columns={col + '_supply_err_rt':'error_rate'
                ,'abs_'+ col + '_supply_err_rt':'abs_error_rate'})
            df_new['method'] = col
            shortage_box = pd.concat([shortage_box, df_new])

    min_date = str(shortage.dropna()['dt'].min().date())
    max_date = str(shortage.dropna()['dt'].max().date())

    fig, ax = plt.subplots(figsize=(20,10))
    fig.suptitle(min_date + ' - ' + max_date + ' Error Rate')
    sns.boxplot(x='fleet', y='error_rate', hue = 'method'
        , data=shortage_box, ax=ax, showfliers=False)
    plt.axhline(y=0, color='red')
    plt.ylabel('Supply Error Rate')
    plt.show()
    plt.close()

    date_365 = str(shortage.dropna()['dt'].max() - datetime.timedelta(days=365))

    fig, ax = plt.subplots(figsize=(20,10))
    sns.boxplot(x='fleet', y='error_rate', hue = 'method'
        , data = shortage_box[(shortage_box.dt>=pd.to_datetime(date_365)) & (shortage.dt<=pd.to_datetime(max_date))]
        , ax=ax, showfliers=False)
    fig.suptitle('Last 365 Days Error Rate')
    plt.ylabel('Supply Error Rate')
    plt.axhline(y=0, color='red')
    plt.show()
    plt.close()

    if date_query:
        for dq in date_query:
            fig, ax = plt.subplots(figsize=(20,10))
            sns.boxplot(x='fleet', y='error_rate', hue = 'method'
            , data=shortage_box.query(dq), ax=ax, showfliers=False)
            fig.suptitle(dq)
            plt.ylabel('Supply Error Rate')
            plt.axhline(y=0, color='red')
            plt.show()
            plt.close()
            
#ECDF plot
pred_test_4['hue'] = pred_test_4['fleet'] + '-' + pred_test_4['seat']
err_col = [a for a in pred_test_4.columns if 'err' in a and 'incr' not in a]
for fleetseat in pred_test_4['hue'].drop_duplicates():
    fig, ax = plt.subplots(figsize = (20,10))
    sns.ecdfplot(data=(pred_test_4[pred_test_4.hue == fleetseat][err_col]), ax = ax)
    #ax.set_yticks(np.arange(0,1.05,.05))
    plt.title(fleetseat)
    plt.grid()
    plt.show()
    plt.close()
    
#decile plotting
decile_box_df = pd.DataFrame()# pred_test_414[pred_test_414.days_prior == 14].copy()
sns.set(font_scale=1.5) 
print(pred_test[['m_demand_err','open_perc_ct_pred_err'
    ,'open_perc_incr_ct_pred_err','open_duty_ct_3dp_pred_err']].mean())
for error in ['m_demand_err','open_perc_ct_pred_err','open_perc_incr_ct_pred_err','open_duty_ct_3dp_pred_err']:#open_perc_3dp_err_rt
    temp_df = pred_test[['fleet','seat','dt',error]].copy()
    temp_df.rename(columns={error:'error'}, inplace=True)
    temp_df['target'] = error
    decile_box_df = pd.concat([decile_box_df, temp_df])
    
decile_box_df['fleet_seat'] = decile_box_df['fleet'] + '-' + decile_box_df['seat']
g = sns.FacetGrid(decile_box_df, row='fleet_seat'#,row='fleet'
                  , height=6, aspect=2, sharex=False)
g.map(sns.boxplot, x='dt', y='error', hue='target',order=sorted(decile_box_df['dt'].drop_duplicates().tolist())#, order=[0,1,2,3,4,5,6,7,8,9]
    , data=decile_box_df, showfliers=False, palette='bright')
g.map(plt.axhline, y=0, ls='--', c='red')
for ax in g.axes.ravel():
    ax.legend()
    ax.set_xlabel('Open Duty Count Decile')
    ax.set_ylabel('Error')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Open Duty Count Decile Error Comparison')
g.fig.tight_layout()
plt.legend()
