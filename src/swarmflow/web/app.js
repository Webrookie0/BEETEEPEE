const $ = id => document.getElementById(id);
const world = $('world'), ctx = world.getContext('2d');
const chart = $('chart'), plot = chart.getContext('2d');
let state = null, playing = false, busy = false, resetting = false, lastStep = 0;
let azimuth = -0.62, elevation = 0.83, zoom = 1, dragging = null, viewMode = 'orbit', panel = null;
let center = [0, 0, 1.4], trails = [], history = [], lastTime = -1;
let queue = Promise.resolve();
let activeTest = null, testCatalog = [], testDefaults = {};
const held = new Set();
const colors = {drone:'#83ddc7', target:'#e6be7c', obstacle:'#dc918b', grid:'#20323e', muted:'#65818e'};
const keyAxis = {w:[1,0,0],s:[-1,0,0],a:[0,1,0],d:[0,-1,0],q:[0,0,1],e:[0,0,-1],
  arrowup:[1,0,0],arrowdown:[-1,0,0],arrowleft:[0,1,0],arrowright:[0,-1,0]};

function notify(message) { $('error').textContent = message; $('error').hidden = !message; }
function api(path, payload) {
  const run = async () => {
    const response = await fetch(path, payload === undefined ? {} : {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)
    });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || `Request failed (${response.status})`);
    return value;
  };
  const next = queue.then(run);
  queue = next.catch(() => {});
  return next;
}

function liveError(s) {
  return Math.sqrt(s.positions.reduce((sum,p,i) => sum+p.reduce((e,x,k)=>e+(x-s.slots[i][k])**2,0),0)/s.positions.length);
}

function accept(s) {
  state = s;
  if (s.time < lastTime) { trails=[]; history=[]; lastTime=-1; }
  if (s.time > lastTime) {
    s.positions.forEach((p,i) => { (trails[i] ||= []).push([...p]); if (trails[i].length>110) trails[i].shift(); });
    history.push([s.time,liveError(s)]);
    history = history.filter(p=>p[0]>=s.time-12);
    lastTime=s.time;
  }
  $('connection').textContent='Local simulation connected';
  $('policy').textContent=s.policy;
  $('learning-note').hidden=s.mode!=='learned';
  syncController(s);
  $('model-note').textContent=`Reduced-order dynamics · illustrative drone attitude · ${s.policy}`;
  $('simulation-time').textContent=`T + ${s.time.toFixed(2).padStart(5,'0')} s`;
  $('slot-error').textContent=liveError(s).toFixed(3);
  $('clearance').textContent=s.metrics.min_surface_clearance_m===null ? '—' : s.metrics.min_surface_clearance_m.toFixed(3);
  const events=s.metrics.events, recovered=events.filter(e=>e.recovery_s!==null).length;
  $('recovered').textContent=`${recovered} / ${events.length}`;
  $('state-note').textContent=`Seed ${s.seed} · ${s.positions.length} agents`;
  $('mission-title').textContent=$('scenario').querySelector(`option[value="${s.scenario}"]`).textContent;
  $('test-badge').hidden=!activeTest;
  if(activeTest)$('test-badge').textContent=`TEST ${activeTest.number} · SEED ${s.seed} · ${s.policy.toUpperCase()}`;
  document.querySelectorAll('.test-card').forEach(card=>card.classList.toggle('active',!!activeTest&&card.dataset.number===String(activeTest.number)));
  const status=s.metrics.failure ? 'Failed' : !s.running ? 'Stopped' : !playing ? 'Paused' : s.obstacles.length ? 'Avoiding' : liveError(s)>.2 ? 'Recovering' : 'Following';
  $('flight-state').textContent=status;
  $('flight-state').style.color=s.metrics.failure ? colors.obstacle : colors.drone;
  $('phase').textContent=s.metrics.failure ? `FLIGHT STOPPED / ${s.metrics.failure.toUpperCase()}` : !s.running ? 'READY' : playing ? 'SIMULATION RUNNING' : 'PAUSED';
  if (!s.running) playing=false;
  $('play').textContent=resetting?'Preparing…':playing?'Stop':'Run';
  $('play').disabled=resetting;
  $('reset').disabled=resetting;
  $('inject').disabled=$('headon').disabled=resetting||!s.running||!playing;
  const pilot=s.scenario==='pilot';
  $('pilot-panel').hidden=!pilot;
  for (const id of ['nudge-forward','nudge-back','nudge-left','nudge-right','nudge-up','nudge-down','nudge-stop'])
    $(id).disabled=resetting||!pilot||!s.running;
}

const controllerHints={
  learned:'Neural network trained here with behavior cloning + PPO. Nothing else runs behind it.',
  baseline:'Hand-coded sampled predictive controller: the reference the network is compared with.',
  slot:'Ablation that flies to its slot and ignores obstacles, showing the obstacles are a real threat.'
};
let shownMode=null,shownCheckpoint=null;
function syncController(s){
  const select=$('controller'),learned=select.querySelector('option[value="learned"]');
  learned.disabled=!s.learned_available;
  learned.textContent=s.learned_available?'Learned policy (trained model)':'Learned policy (no trained model yet)';
  if(s.mode!==shownMode){select.value=s.mode||'baseline';shownMode=s.mode;}
  select.disabled=resetting;
  $('controller-hint').textContent=controllerHints[s.mode]||'';
  $('learning-checkpoint').textContent=s.checkpoint||'';
  if(s.checkpoint!==shownCheckpoint){shownCheckpoint=s.checkpoint;if(modelReport)loadModel();}
}

function steerVelocity() {
  const v=[0,0,0];
  for (const key of held) {
    const axis=keyAxis[key];
    if(axis){v[0]+=axis[0];v[1]+=axis[1];v[2]+=axis[2];}
  }
  const norm=Math.hypot(v[0],v[1],v[2]);
  if(norm>1e-6){v[0]=1.2*v[0]/norm;v[1]=1.2*v[1]/norm;v[2]=1.2*v[2]/norm;}
  return v;
}

async function steer(payload) {
  if(!state||state.scenario!=='pilot'||!state.running)return;
  try {notify('');accept(await api('/api/steer',payload));}
  catch(e){notify(e.message);}
}

async function reset(start=false, continuous=false) {
  if(resetting)return;
  resetting=true;
  activeTest=null;
  playing=false;
  const scenario=$('scenario').value;
  const geometry=$('geometry').value;
  const world=$('terrain').value;
  const spacing=Number($('compaction').value);
  const drones=parseInt($('drones').value,10);
  const seed=parseInt($('seed').value,10)||0;
  const controller=$('controller').value;
  if(state)accept(state);
  try {
    notify('');
    const s=await api('/api/reset',{
      scenario, drones, seed, geometry, world, spacing, controller,
      continuous: !!continuous || !!start,
      duration: (continuous || start) ? 3600 : 24
    });
    trails=[];history=[];lastTime=-1;
    center=[s.target[0]-2,0,1.4];
    $('geometry').value=s.geometry||geometry;
    $('terrain').value=s.world||world;
    $('scenario').value=s.scenario||scenario;
    if(s.config&&s.config.spacing!=null){
      const options=[...$('compaction').options].map(o=>Number(o.value));
      const nearest=options.reduce((a,b)=>Math.abs(b-s.config.spacing)<Math.abs(a-s.config.spacing)?b:a);
      $('compaction').value=String(nearest);
    }
    playing=start;
    accept(s);
  } catch(e) {notify(e.message);if(state)$('controller').value=state.mode;}
  finally {resetting=false;if(state)accept(state);}
}
$('reset').onclick=()=>reset(false, false);
$('controller').onchange=()=>reset(playing, playing);
$('geometry').onchange=()=>reset(playing, playing);
$('compaction').onchange=()=>reset(playing, playing);
$('terrain').onchange=()=>reset(playing, playing);
$('scenario').onchange=()=>{
  scenarioHint();
  if($('scenario').value==='terrain_follow')$('terrain').value='hills';
  reset(playing, playing);
};
$('play').onclick=()=>{
  if(!state||resetting)return;
  if(playing){playing=false;accept(state);return;}
  // Resume a live continuous flight; only rebuild when idle or failed.
  if(state.running && !state.metrics.failure){playing=true;accept(state);return;}
  reset(true, true);
};
async function inject(angle) {
  try {notify('');accept(await api('/api/inject',{speed:Number($('speed').value),radius:Number($('radius').value),angle}));}
  catch(e){notify(e.message);}
}
$('inject').onclick=()=>inject(Math.PI/2);
$('headon').onclick=()=>inject(Math.PI);
const nudges={
  'nudge-forward':[1.2,0,0],'nudge-back':[-1.2,0,0],'nudge-left':[0,1.2,0],'nudge-right':[0,-1.2,0],
  'nudge-up':[0,0,1.0],'nudge-down':[0,0,-1.0],'nudge-stop':[0,0,0]
};
for (const [id,velocity] of Object.entries(nudges)) $(id).onclick=()=>steer({velocity});
window.addEventListener('keydown',e=>{
  const key=e.key.toLowerCase();
  if(!(key in keyAxis)||!state||state.scenario!=='pilot')return;
  if(['input','select','textarea'].includes((e.target.tagName||'').toLowerCase()))return;
  e.preventDefault();held.add(key);
});
window.addEventListener('keyup',e=>{
  const key=e.key.toLowerCase();
  if(key in keyAxis){held.delete(key);if(state&&state.scenario==='pilot')steer({velocity:steerVelocity()});}
});
for (const [id,unit] of [['speed','m/s'],['radius','m']]) $(id).oninput=()=>{$(`${id}-value`).textContent=`${Number($(id).value).toFixed(1)} ${unit}`;};
function setView(mode){
  viewMode=mode;
  $('orbit').classList.toggle('selected', mode==='orbit');
  $('axis').classList.toggle('selected', mode==='axis');
  $('view-hint').textContent=mode==='axis'
    ? 'METERS · XY / XZ / YZ SIDE VIEWS · SCROLL TO ZOOM'
    : 'METERS · DRAG TO ORBIT · SCROLL TO ZOOM';
}
$('axis').onclick=()=>setView('axis');
$('orbit').onclick=()=>{elevation=.83;azimuth=-.62;setView('orbit');};
$('export').onclick=()=>{
  if(!state)return;
  const output=document.createElement('canvas');output.width=world.width;output.height=world.height;
  const brush=output.getContext('2d');brush.fillStyle='#0e161c';brush.fillRect(0,0,output.width,output.height);brush.drawImage(world,0,0);
  const ratio=window.devicePixelRatio||1;brush.scale(ratio,ratio);brush.fillStyle='#e5edee';brush.font='12px monospace';
  brush.fillText(`SwarmFlow / ${state.scenario} / ${state.policy} / seed ${state.seed} / ${state.time.toFixed(2)} s`,18,24);
  brush.fillStyle='#889ba4';brush.font='10px monospace';brush.fillText('3D stabilized-flight model · illustrative attitude',18,42);
  output.toBlob(blob=>{const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=`swarmflow-${state.scenario}.png`;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);});
};
world.onpointerdown=e=>{if(viewMode!=='orbit')return;dragging=[e.clientX,e.clientY];world.setPointerCapture(e.pointerId);};
world.onpointermove=e=>{if(dragging){azimuth+=(e.clientX-dragging[0])*.006;elevation=Math.max(.15,Math.min(1.57,elevation+(e.clientY-dragging[1])*.006));dragging=[e.clientX,e.clientY];}};
world.onpointerup=()=>dragging=null;
world.onwheel=e=>{e.preventDefault();zoom=Math.max(.45,Math.min(2,zoom*Math.exp(-e.deltaY*.001)));};

function fit(canvas, context) {
  const ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;
  if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}
  context.setTransform(ratio,0,0,ratio,0,0);
  return [w,h];
}
let w=0,h=0,scale=1;
function project(p){
  const x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];
  if(panel){
    let u,v,depth;
    if(panel.map==='xy'){u=x;v=-y;depth=z;}
    else if(panel.map==='xz'){u=x;v=-z;depth=y;}
    else {u=y;v=-z;depth=x;}
    return [panel.ox+panel.pw*.5+u*scale, panel.oy+panel.ph*.55+v*scale, depth];
  }
  const u=x*Math.cos(azimuth)-y*Math.sin(azimuth),v=x*Math.sin(azimuth)+y*Math.cos(azimuth);
  return [w*.5+u*scale,h*.53+(v*Math.sin(elevation)-z*Math.cos(elevation))*scale,v*Math.cos(elevation)+z*Math.sin(elevation)];
}
function line(a,b,color,width=1,dash=[]){const p=project(a),q=project(b);ctx.beginPath();ctx.setLineDash(dash);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.setLineDash([]);}
function circle(p,r,fill,stroke){ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}}
function drawDrone(p,index,t){
  const a=state.accelerations[index], tiltX=-a[0]*.02, tiltY=-a[1]*.02, arm=.14;
  const rotorPoints=[[-arm,-arm],[arm,arm],[-arm,arm],[arm,-arm]].map(([x,y])=>[p[0]+x,p[1]+y,p[2]+tiltX*x+tiltY*y]);
  line(rotorPoints[0],rotorPoints[1],colors.drone,1.6);line(rotorPoints[2],rotorPoints[3],colors.drone,1.6);
  for(const r of rotorPoints){
    const point=project(r);circle(point,Math.max(2,scale*.08),'#18332f',colors.drone);
    const theta=t*.045+index,rr=scale*.06;
    ctx.beginPath();ctx.moveTo(point[0]+Math.cos(theta)*rr,point[1]+Math.sin(theta)*rr*.4);ctx.lineTo(point[0]-Math.cos(theta)*rr,point[1]-Math.sin(theta)*rr*.4);ctx.strokeStyle='#83ddc777';ctx.stroke();
  }
  const point=project(p);circle(point,Math.max(1.8,scale*.06),colors.drone);
  ctx.fillStyle='#bedbd8';ctx.font='9px ui-monospace,monospace';ctx.fillText(String(index+1).padStart(2,'0'),point[0]+8,point[1]-8);
}
function drawObject(obj){
  const p=project(obj.position),r=obj.radius*scale;
  circle(p,r,'#dc918b27',colors.obstacle);
  ctx.strokeStyle='#dc918b77';ctx.beginPath();ctx.ellipse(p[0],p[1],r,r*.3,0,0,Math.PI*2);ctx.stroke();
  ctx.beginPath();ctx.ellipse(p[0],p[1],r*.35,r,0,0,Math.PI*2);ctx.stroke();
  const end=obj.position.map((v,k)=>v+obj.velocity[k]*.7);line(obj.position,end,'#dc918b88',1,[3,3]);
}
function groundAt(x,y,hills,peaks){
  if(!hills)return 0;
  let h=1.15*Math.sin(0.55*x)*(0.7+0.3*Math.cos(0.3*y))
    +0.75*Math.sin(0.33*x+1.0)
    +0.5*Math.sin(0.95*x)*Math.cos(0.45*y)
    +0.28*Math.sin(0.2*x-0.28*y);
  for(const p of peaks){const r=((x-p.x)**2+(y-p.y)**2)/(2*Math.max(p.width*p.width,1e-6));h+=p.height*Math.exp(-r);}
  return Math.max(0,h);
}
function drawScene(t){
  const baseX=Math.floor(center[0]),baseY=Math.floor(center[1]);
  const hills=state.world==='hills';
  const peaks=state.peaks||[];
  const ground=(x,y)=>groundAt(x,y,hills,peaks);
  for(let i=-14;i<=14;i++){
    if(hills){
      for(let j=-13;j<14;j++){
        const major=j%5===0?'#2a424f':colors.grid;
        line([baseX+i,baseY+j,ground(baseX+i,baseY+j)],[baseX+i,baseY+j+1,ground(baseX+i,baseY+j+1)],major,.7);
        line([baseX+j,baseY+i,ground(baseX+j,baseY+i)],[baseX+j+1,baseY+i,ground(baseX+j+1,baseY+i)],major,.7);
      }
    } else {
      line([baseX+i,baseY-14,0],[baseX+i,baseY+14,0],i%5===0?'#2a424f':colors.grid,.7);
      line([baseX-14,baseY+i,0],[baseX+14,baseY+i,0],i%5===0?'#2a424f':colors.grid,.7);
    }
  }
  trails.forEach(points=>{for(let k=1;k<points.length;k++)line(points[k-1],points[k],'#80d4c226',1);});
  state.slots.forEach((p,i)=>{
    const q=project(p);ctx.strokeStyle='#819da378';ctx.lineWidth=1;ctx.strokeRect(q[0]-4,q[1]-4,8,8);
    line(p,state.positions[i],'#90beb347',.8,[3,4]);
    const shadow=project([state.positions[i][0],state.positions[i][1],ground(state.positions[i][0],state.positions[i][1])]);circle(shadow,scale*.17,'#70b9b211');
  });
  const tp=project(state.target);ctx.save();ctx.translate(tp[0],tp[1]);ctx.rotate(Math.PI/4);ctx.fillStyle='#e6be7c20';ctx.strokeStyle=colors.target;ctx.lineWidth=1.4;ctx.fillRect(-6,-6,12,12);ctx.strokeRect(-6,-6,12,12);ctx.restore();
  ctx.fillStyle=colors.target;ctx.font='9px ui-monospace,monospace';ctx.fillText('REFERENCE',tp[0]+13,tp[1]+3);
  let fx=1,fy=0,fz=0;
  const tv=state.target_velocity||[0,0,0];
  const spd=Math.hypot(tv[0],tv[1],tv[2]);
  if(spd>0.05){fx=tv[0]/spd;fy=tv[1]/spd;fz=tv[2]/spd;}
  const tip=[state.target[0]+0.7*fx,state.target[1]+0.7*fy,state.target[2]+0.7*fz];
  const left=[state.target[0]+0.48*fx-0.12*fy,state.target[1]+0.48*fy+0.12*fx,state.target[2]+0.48*fz];
  const right=[state.target[0]+0.48*fx+0.12*fy,state.target[1]+0.48*fy-0.12*fx,state.target[2]+0.48*fz];
  line(state.target,tip,colors.target,1.6);
  line(left,tip,colors.target,1.4);line(right,tip,colors.target,1.4);
  const tip2d=project(tip);
  ctx.fillStyle=colors.target;ctx.font='8px ui-monospace,monospace';ctx.fillText('FWD',tip2d[0]+5,tip2d[1]-1);
  const objects=state.positions.map((p,i)=>({depth:project(p)[2],draw:()=>drawDrone(p,i,t)}));
  state.obstacles.forEach(o=>objects.push({depth:project(o.position)[2],draw:()=>drawObject(o)}));
  objects.sort((a,b)=>a.depth-b.depth).forEach(o=>o.draw());
}
function draw(t){
  [w,h]=fit(world,ctx);ctx.clearRect(0,0,w,h);if(!state)return;
  center[0]+=(state.target[0]-2-center[0])*.04;
  center[1]+=(state.target[1]-center[1])*.04;
  if(viewMode==='axis'){
    const gap=8, labels=[['xz','XZ · SIDE'],['xy','XY · TOP'],['yz','YZ · SIDE']];
    const pw=(w-gap*4)/3, ph=h-16;
    labels.forEach(([map,label],i)=>{
      panel={ox:gap+i*(pw+gap), oy:8, pw, ph, map};
      scale=Math.min(pw/12, ph/9)*zoom;
      ctx.save();
      ctx.beginPath();ctx.rect(panel.ox, panel.oy, panel.pw, panel.ph);ctx.clip();
      ctx.fillStyle='#0e161c';ctx.fillRect(panel.ox, panel.oy, panel.pw, panel.ph);
      drawScene(t);
      ctx.restore();
      ctx.strokeStyle='#2a424f';ctx.lineWidth=1;ctx.strokeRect(panel.ox+.5, panel.oy+.5, panel.pw-1, panel.ph-1);
      ctx.fillStyle=colors.muted;ctx.font='10px ui-monospace,monospace';
      ctx.fillText(label, panel.ox+10, panel.oy+16);
    });
    panel=null;
  } else {
    scale=Math.min(w/17,h/10)*zoom;
    drawScene(t);
  }
  drawChart();
}
function drawChart(){
  const [cw,ch]=fit(chart,plot);plot.clearRect(0,0,cw,ch);if(!history.length)return;
  const max=Math.max(.5,...history.map(p=>p[1]))*1.1,left=30,right=cw-8,bottom=ch-16,top=7;
  plot.font='8px ui-monospace,monospace';plot.fillStyle=colors.muted;plot.strokeStyle=colors.grid;plot.lineWidth=1;
  for(const y of [0,max/2,max]){const yy=bottom-y/max*(bottom-top);plot.beginPath();plot.moveTo(left,yy);plot.lineTo(right,yy);plot.stroke();plot.fillText(y.toFixed(1),0,yy+3);}
  const start=Math.max(0,state.time-12),end=Math.max(12,state.time);
  plot.fillText(`${start.toFixed(0)} s`,left,ch-1);plot.fillText(`${end.toFixed(0)} s`,right-20,ch-1);
  plot.beginPath();history.forEach((p,i)=>{const x=left+(p[0]-start)/(end-start)*(right-left),y=bottom-p[1]/max*(bottom-top);i?plot.lineTo(x,y):plot.moveTo(x,y);});plot.strokeStyle=colors.drone;plot.lineWidth=1.5;plot.stroke();
}
async function tick(t){
  draw(t);
  if(playing&&!busy&&t-lastStep>=95){
    busy=true;lastStep=t;
    try{
      if(state&&state.scenario==='pilot'&&held.size)await api('/api/steer',{velocity:steerVelocity()});
      const next=await api('/api/step',{steps:2});
      accept(next);
      if(!next.running)playing=false;
    }
    catch(e){playing=false;if(state)accept(state);notify(e.message);$('connection').textContent='Connection interrupted';}
    finally{busy=false;}
  }
  requestAnimationFrame(tick);
}
function scenarioLabel(value){
  const option=$('scenario').querySelector(`option[value="${value}"]`);
  return option?option.textContent:value;
}
function el(tag, className, text){
  const node=document.createElement(tag);
  if(className)node.className=className;
  if(text!==undefined)node.textContent=text;
  return node;
}
function scenarioHint(){
  const hint=$('scenario-tests'), scenario=$('scenario').value;
  hint.replaceChildren();
  const users=testCatalog.filter(t=>t.scenarios.includes(scenario));
  if(!users.length){hint.textContent=testCatalog.length?'Not part of a numbered test (free exploration).':'';return;}
  hint.append('Used in tests ');
  users.forEach(t=>{const a=el('a','',String(t.number));a.href=`#test-${t.number}`;a.title=t.title;hint.append(a);});
}
async function watch(test, scenario){
  if(resetting||!test.setup)return;
  resetting=true;playing=false;
  const setup=test.setup;
  const payload={scenario, drones:setup.drones, seed:setup.seed, geometry:testDefaults.geometry,
    world:scenario==='terrain_follow'?'hills':testDefaults.world, spacing:testDefaults.spacing,
    continuous:false, duration:24, controller:setup.controller||'baseline'};
  for(const key of ['noise','dropout','obstacles'])if(setup[key]!==undefined)payload[key]=setup[key];
  try{
    notify('');
    const s=await api('/api/reset',payload);
    trails=[];history=[];lastTime=-1;center=[s.target[0]-2,0,1.4];
    $('scenario').value=s.scenario;$('drones').value=String(s.positions.length);$('seed').value=s.seed;
    $('geometry').value=s.geometry||payload.geometry;$('terrain').value=s.world||payload.world;
    const spacings=[...$('compaction').options].map(o=>Number(o.value));
    $('compaction').value=String(spacings.reduce((a,b)=>Math.abs(b-s.config.spacing)<Math.abs(a-s.config.spacing)?b:a));
    activeTest=test;playing=true;scenarioHint();accept(s);
    document.querySelector('.flight-panel').scrollIntoView({behavior:'smooth',block:'center'});
  }catch(e){notify(e.message);}
  finally{resetting=false;if(state)accept(state);}
}
function renderTests(){
  const list=$('test-list');list.replaceChildren();
  for(const test of testCatalog){
    const card=el('article','test-card');card.id=`test-${test.number}`;card.dataset.number=String(test.number);
    const top=el('div','test-top');
    top.append(el('span','test-num',String(test.number).padStart(2,'0')),el('span','test-title',test.title));
    const last=test.last;
    const verdict=el('span',`verdict ${last?last.verdict.toLowerCase():''}`,last?`LAST RUN: ${last.verdict}`:'NOT RUN YET');
    if(last)verdict.title=`${last.finished} · ${last.elapsed_s} s${last.report?` · ${last.report}`:''}\n${last.notes.join('\n')}`;
    top.append(verdict);card.append(top);
    const what=el('p');what.append(el('b','','What happens: '),test.purpose);card.append(what);
    const pass=el('p');pass.append(el('b','','Passes when: '),test.criterion);card.append(pass);
    if(test.scenarios.length){
      const chips=el('div','chips');chips.append('Scenarios:');
      for(const scenario of test.scenarios){
        const chip=el(test.setup?'button':'span','chip',scenarioLabel(scenario));
        if(chip.tagName==='BUTTON'){chip.type='button';chip.title=`Watch test ${test.number} on this scenario`;chip.onclick=()=>watch(test,scenario);}
        chips.append(chip);
      }
      card.append(chips);
    }
    const actions=el('div','test-actions');
    if(test.setup){
      const button=el('button','watch',`▶ Watch test ${test.number}`);button.type='button';
      button.title=`Fly ${scenarioLabel(test.scenarios[0])} with this test's setup`;
      button.onclick=()=>watch(test,test.scenarios[0]);
      actions.append(button);
    }
    const run=el('small');run.append('Run: ',el('code','',test.command),` · ${test.estimate} · ${test.source}`);
    actions.append(run);card.append(actions);
    if(test.watch){const note=el('p');note.append(el('b','','Watch the trained policy: '),el('code','',test.watch));card.append(note);}
    else if(!test.setup){card.append(el('p','','Code-level checks only: nothing to fly. Results appear in the terminal.'));}
    list.append(card);
  }
  scenarioHint();
}
async function loadTests(){
  try{const data=await api('/api/tests');testCatalog=data.tests;testDefaults=data.defaults;renderTests();}
  catch(e){$('test-list').replaceChildren(el('p','hint',`Could not load the test catalog: ${e.message}`));}
}
$('tests-refresh').onclick=loadTests;

let modelReport=null;
const stageColors=['#79d9bf','#e7bd79','#9fb7ff','#ee9389'];
const num=(n,d=3)=>n===null||n===undefined?'—':Number(n).toFixed(d);
const count=n=>n===null||n===undefined?'—':Number(n).toLocaleString();
const pct=n=>n===null||n===undefined?'—':`${Math.round(n*100)}%`;
async function loadModel(){
  try{modelReport=await api('/api/model');}
  catch(e){modelReport={available:false,error:e.message};}
  renderModel();
}
$('model-refresh').onclick=loadModel;
const clock=seconds=>{const s=Math.max(0,Math.round(seconds));return `${Math.floor(s/3600)}:${String(Math.floor(s%3600/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;};
function renderTraining(t,hasModel){
  const box=$('training-live');
  if(!t||(!t.active&&hasModel)){box.hidden=true;return;}
  box.hidden=false;
  const done=t.progress===null||t.progress===undefined?0:Math.max(0,Math.min(1,t.progress));
  $('training-bar').style.width=`${(done*100).toFixed(1)}%`;
  $('training-title').textContent=t.active?`Training in progress · stage ${t.stage} of ${t.stages}`
    :`Last training run · stage ${t.stage} of ${t.stages} · ${t.finished?'finished':'stopped'}`;
  const parts=[t.phase];
  if(t.phase==='behavior cloning')parts.push(`${t.cloning_rounds_done} cloning round(s) done`);
  if(t.target_environment_steps)parts.push(`${count(t.environment_steps-t.start_environment_steps)} / ${count(t.target_environment_steps-t.start_environment_steps)} PPO steps (${pct(t.progress)})`);
  if(t.steps_per_second)parts.push(`${count(Math.round(t.steps_per_second))} steps/s`);
  if(t.eta_seconds!==null&&t.eta_seconds!==undefined)parts.push(`about ${clock(t.eta_seconds)} left in this stage`);
  if(t.recent_failure_rate!==null&&t.recent_failure_rate!==undefined)parts.push(`failed episodes (last 100) ${pct(t.recent_failure_rate)}`);
  if(t.last_score!==null&&t.last_score!==undefined)parts.push(`last validation ${num(t.last_score)} · best ${num(t.best_score)}`);
  if(!t.active&&!t.finished)parts.push(`no update for ${clock(t.seconds_since_update)}; run make training again to continue`);
  $('training-text').textContent=`${t.run} · ${parts.join(' · ')}`;
}
function renderModel(){
  const r=modelReport,facts=$('model-facts'),checks=$('model-checks'),verdict=$('model-verdict');
  facts.replaceChildren();checks.replaceChildren();$('model-table').replaceChildren();
  verdict.className='verdict';
  renderTraining(r&&r.training,r&&r.available);
  if(!r||!r.available){
    const live=r&&r.training&&r.training.active;
    verdict.textContent=live?'TRAINING…':'NO MODEL YET';
    $('model-description').textContent=live
      ?'Training is running. The first model appears here after behavior cloning and the first validation (a few minutes); the viewer then switches to it automatically.'
      :'No trained model yet. Start training with make training (or make run train to watch it here); the viewer picks up the best model automatically.';
    checks.append(el('li','pending',live?'Waiting for the first validated model…':'Train the model first.'));
    drawCurves();return;
  }
  if(r.error){
    verdict.textContent='UNREADABLE';
    $('model-description').textContent=`Could not read the training records of ${r.checkpoint}: ${r.error}`;
    drawCurves();return;
  }
  const test=r.test;
  if(test){verdict.classList.add(test.learned_correctly?'pass':'fail');verdict.textContent=test.learned_correctly?'LEARNED CORRECTLY':'NOT LEARNED YET';}
  else verdict.textContent='NOT TESTED YET';
  const training=r.training&&r.training.active;
  $('model-description').textContent=test
    ?`Loaded ${r.checkpoint}. The verdict comes from ${test.test_scenarios.length} scenarios × ${test.test_seeds[1]-test.test_seeds[0]+1} held-out seeds (${test.test_seeds[0]}–${test.test_seeds[1]}), never used for training or for choosing the checkpoint.`
    :training
      ?`Showing the best model so far: ${r.checkpoint}. It is replaced automatically whenever training saves a better one; the held-out verdict appears when the whole pipeline finishes.`
      :`Loaded ${r.checkpoint}. Run make showcase (python -m swarmflow showcase) to test it on held-out seeds against the baseline and get a verdict.`;
  const add=(key,value)=>facts.append(el('dt','',key),el('dd','',value));
  const total=r.pipeline_steps||r.environment_steps;
  add('Trained for',`${count(total)} PPO environment steps over ${r.stages.length||1} stage(s) · ${count(total*(r.drones||8))} drone transitions`);
  add('Algorithm',r.algorithm||'—');
  if(r.stages.length)add('Curriculum',r.stages.map(s=>`${s.label}: ${(s.scenarios||[]).join(', ')}${s.finished?'':' (unfinished)'}`).join('  →  '));
  const cloning=r.stages.flatMap(s=>s.cloning||[]);
  add('Warm start',cloning.length?`behavior cloning from the predictive controller, ${cloning.length} round(s) incl. DAgger, ${count(cloning[cloning.length-1].agent_samples)} labelled samples`:'none: PPO from random weights');
  if(r.selection)add('Chosen checkpoint',`step ${count(r.selection.environment_steps)} · validation score ${num(r.selection.score)} · success ${pct(r.selection.success_rate)} · slot RMSE ${num(r.selection.mean_slot_rmse_m)} m`);
  if(test)add('Held-out test score',`learned ${num(test.test_score_learned)}  vs  baseline ${num(test.test_score_baseline)}`);
  if(test)for(const c of test.checks){
    const li=el('li',c.passed?'pass':'fail');
    li.append(el('b','',c.passed?'PASS':'FAIL'),el('span','',c.name),el('small','',c.detail));checks.append(li);
  }
  else checks.append(el('li','pending','Not tested on held-out seeds yet. Run make showcase.'));
  if(test)renderModelTable(test.scenarios);
  drawCurves();
}
function renderModelTable(rows){
  const table=el('table'),head=el('tr');
  for(const title of ['Scenario','Success · baseline','Success · learned','Slot RMSE · baseline','Slot RMSE · learned','Recovered · baseline','Recovered · learned','Min clearance · learned'])head.append(el('th','',title));
  table.append(head);
  for(const row of rows){
    const b=row.baseline,l=row.learned,tr=el('tr');
    const formation=['hold','follow','turn'].includes(row.scenario);
    const cells=[[scenarioLabel(row.scenario)],[pct(b.success_rate)],[pct(l.success_rate),l.success_rate<1],
      [`${num(b.mean_slot_rmse_m)} m`],[`${num(l.mean_slot_rmse_m)} m`,formation&&l.mean_slot_rmse_m>=.35],
      [b.events?`${b.recovered_events}/${b.events}`:'—'],[l.events?`${l.recovered_events}/${l.events}`:'—',l.recovered_events<l.events],
      [l.min_surface_clearance_m===null?'—':`${num(l.min_surface_clearance_m)} m`]];
    for(const [text,bad] of cells)tr.append(el('td',bad?'bad':'',text));
    table.append(tr);
  }
  $('model-table').append(table);
}
function plotSeries(canvas,series,options){
  const g=canvas.getContext('2d'),[cw,ch]=fit(canvas,g);g.clearRect(0,0,cw,ch);
  g.font='9px ui-monospace,monospace';g.fillStyle=colors.muted;
  const points=series.flatMap(s=>s.points);
  if(!points.length){g.fillText('No training history for this model yet',8,ch/2);return;}
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),refs=options.refs||[];
  const x0=Math.min(...xs),x1=Math.max(x0+1,...xs);
  let y0=options.fixed?options.fixed[0]:Math.min(...ys,...refs),y1=options.fixed?options.fixed[1]:Math.max(...ys,...refs);
  if(!options.fixed){const pad=(y1-y0||1)*.12;y0-=pad;y1+=pad;}
  const left=40,right=cw-8,top=14,bottom=ch-18;
  const X=x=>left+(x-x0)/(x1-x0)*(right-left),Y=y=>bottom-(y-y0)/(y1-y0)*(bottom-top);
  g.lineWidth=1;g.strokeStyle=colors.grid;
  for(const v of [y0,(y0+y1)/2,y1]){g.beginPath();g.moveTo(left,Y(v));g.lineTo(right,Y(v));g.stroke();g.fillText(options.label(v),2,Y(v)+3);}
  g.strokeStyle='#e7bd7988';g.setLineDash([4,4]);
  for(const v of refs){g.beginPath();g.moveTo(left,Y(v));g.lineTo(right,Y(v));g.stroke();}
  g.setLineDash([]);
  g.fillText(count(x0),left,ch-4);const end=`${count(x1)} steps`;g.fillText(end,right-g.measureText(end).width,ch-4);
  series.forEach((s,i)=>{
    if(!s.points.length)return;
    const color=stageColors[i%stageColors.length],start=X(s.points[0][0]);
    if(i>0){g.strokeStyle='#65818e';g.setLineDash([2,3]);g.beginPath();g.moveTo(start,top);g.lineTo(start,bottom);g.stroke();g.setLineDash([]);}
    g.strokeStyle=color;g.lineWidth=1.6;g.beginPath();
    s.points.forEach((p,k)=>k?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));g.stroke();
    if(options.markers)for(const p of s.points){g.fillStyle=color;g.beginPath();g.arc(X(p[0]),Y(p[1]),2.2,0,Math.PI*2);g.fill();}
    g.fillStyle=color;g.fillText(s.name,start+4,10);g.fillStyle=colors.muted;
  });
  if(options.best!==null&&options.best!==undefined){
    const owner=[...series].reverse().find(s=>s.points.some(p=>p[2]===options.best));
    const hit=owner&&owner.points.find(p=>p[2]===options.best);
    if(hit){const x=X(hit[0]),y=Y(hit[1]);g.fillStyle=colors.target;g.beginPath();g.moveTo(x,y-6);g.lineTo(x+6,y);g.lineTo(x,y+6);g.lineTo(x-6,y);g.closePath();g.fill();}
  }
}
function drawCurves(){
  const r=modelReport,stages=(r&&r.available&&r.stages)||[];
  const best=r&&r.selection?r.selection.environment_steps:null;
  const series=pick=>stages.map(s=>({name:s.label,points:pick(s)}));
  const x=row=>row.x??row.environment_steps;
  plotSeries($('curve-score'),series(s=>s.evaluations.filter(e=>e.score!==null).map(e=>[x(e),e.score,e.environment_steps])),
    {label:v=>v.toFixed(2),refs:[1],markers:true,best});
  plotSeries($('curve-failure'),series(s=>s.updates.filter(u=>u.recent_failure_rate!==null).map(u=>[x(u),u.recent_failure_rate,u.environment_steps])),
    {label:v=>`${Math.round(v*100)}%`,fixed:[0,1]});
  plotSeries($('curve-rmse'),series(s=>s.updates.filter(u=>u.recent_slot_rmse_m!==null).map(u=>[x(u),u.recent_slot_rmse_m,u.environment_steps])),
    {label:v=>v.toFixed(2),refs:[.35]});
}
window.addEventListener('resize',drawCurves);
setInterval(()=>{
  const t=modelReport&&modelReport.training;
  if(modelReport&&modelReport.available&&!(t&&t.active))return;
  loadModel();
  if(!playing&&!resetting)api('/api/state').then(accept).catch(()=>{});
},10000);

api('/api/state').then(s=>{
  $('scenario').value=s.scenario;$('drones').value=String(s.positions.length);$('seed').value=s.seed;
  if(s.geometry)$('geometry').value=s.geometry;
  center=[s.target[0]-2,0,1.4];accept(s);
  loadTests();
  loadModel();
}).catch(e=>notify(`Could not connect to the simulator: ${e.message}. Keep the Python server running and reload.`));
requestAnimationFrame(tick);
