from django.shortcuts import render

from . import engine

def predict_data(request):
    if request.method == 'POST':
        t2mdew = float(request.POST.get('t2mdew'))
        t2m = float(request.POST.get('t2m'))
        ps = float(request.POST.get('ps'))
        tqv = float(request.POST.get('tqv'))
        tql = float(request.POST.get('tql'))
        h1000 = float(request.POST.get('h1000'))
        disph = float(request.POST.get('disph'))
        frcan = float(request.POST.get('frcan'))
        hlml = float(request.POST.get('hlml'))
        rhoa = float(request.POST.get('rhoa'))
        cig = float(request.POST.get('cig'))
        ws = float(request.POST.get('ws'))
        cldcr = float(request.POST.get('cldcr'))
        v_2m = float(request.POST.get('v_2m'))
        v_50m = float(request.POST.get('v_50m'))
        v_850 = float(request.POST.get('v_850'))
        result = engine.predict(t2mdew, t2m, ps, tqv, tql, h1000, disph, frcan, hlml, rhoa, cig, ws, cldcr, v_2m, v_50m, v_850)
        context = {
            'result': result,
            't2mdew': t2mdew,
            't2m': t2m,
            'ps': ps,
            'tqv': tqv,
            'tql': tql,
            'h1000': h1000,
            'disph': disph,
            'frcan': frcan,
            'hlml': hlml,
            'rhoa': rhoa,
            'cig': cig,
            'ws': ws,
            'cldcr': cldcr,
            'v_2m': v_2m,
            'v_50m': v_50m,
            'v_850': v_850
        }
        return render(request, 'index.html', context)
    return render(request, 'index.html')
