import{j as e}from"./jsx-runtime-Bywvkw1S.js";import{C as n}from"./ConfidenceIndicator-CRl75YCL.js";import"./index-CleY8y_P.js";import"./_commonjsHelpers-Cpj98o6Y.js";import"./cn-CytzSlOG.js";const fe={title:"Components/ConfidenceIndicator",component:n,parameters:{layout:"centered",docs:{description:{component:`The ConfidenceIndicator component displays a confidence score with color-coded styling
and an optional click handler for showing detailed explanations.

## Color Coding
- **Green** (>0.8): High confidence
- **Yellow** (0.5-0.8): Medium confidence
- **Red** (<0.5): Low confidence

## Accessibility
- ARIA labels for screen readers
- Keyboard navigable (Tab, Enter, Space)
- Color-blind friendly with text labels`}}},tags:["autodocs"],argTypes:{confidence:{control:{type:"range",min:0,max:1,step:.01},description:"Confidence score between 0 and 1"},onClick:{action:"clicked",description:"Optional click handler for showing explanation modal"}}},o={args:{confidence:.92}},c={args:{confidence:.65}},r={args:{confidence:.35}},s={args:{confidence:.85,onClick:()=>alert("Confidence explanation modal would open here")}},i={args:{confidence:.8}},a={args:{confidence:.5}},d={args:{confidence:1}},t={args:{confidence:.05}},p={args:{confidence:.75},render:()=>e.jsxs("div",{className:"flex flex-col gap-4",children:[e.jsx(n,{confidence:.95}),e.jsx(n,{confidence:.85}),e.jsx(n,{confidence:.75}),e.jsx(n,{confidence:.65}),e.jsx(n,{confidence:.55}),e.jsx(n,{confidence:.45}),e.jsx(n,{confidence:.35}),e.jsx(n,{confidence:.25}),e.jsx(n,{confidence:.15})]})},l={args:{confidence:.75,className:"shadow-lg"}};var m,f,u,g,C;o.parameters={...o.parameters,docs:{...(m=o.parameters)==null?void 0:m.docs,source:{originalSource:`{
  args: {
    confidence: 0.92
  }
}`,...(u=(f=o.parameters)==null?void 0:f.docs)==null?void 0:u.source},description:{story:"High confidence indicator (>0.8) with green styling",...(C=(g=o.parameters)==null?void 0:g.docs)==null?void 0:C.description}}};var y,h,w,x,I;c.parameters={...c.parameters,docs:{...(y=c.parameters)==null?void 0:y.docs,source:{originalSource:`{
  args: {
    confidence: 0.65
  }
}`,...(w=(h=c.parameters)==null?void 0:h.docs)==null?void 0:w.source},description:{story:"Medium confidence indicator (0.5-0.8) with yellow styling",...(I=(x=c.parameters)==null?void 0:x.docs)==null?void 0:I.description}}};var b,j,S,k,L;r.parameters={...r.parameters,docs:{...(b=r.parameters)==null?void 0:b.docs,source:{originalSource:`{
  args: {
    confidence: 0.35
  }
}`,...(S=(j=r.parameters)==null?void 0:j.docs)==null?void 0:S.source},description:{story:"Low confidence indicator (<0.5) with red styling",...(L=(k=r.parameters)==null?void 0:k.docs)==null?void 0:L.description}}};var v,M,B,H,N;s.parameters={...s.parameters,docs:{...(v=s.parameters)==null?void 0:v.docs,source:{originalSource:`{
  args: {
    confidence: 0.85,
    onClick: () => alert('Confidence explanation modal would open here')
  }
}`,...(B=(M=s.parameters)==null?void 0:M.docs)==null?void 0:B.source},description:{story:"Clickable indicator with onClick handler",...(N=(H=s.parameters)==null?void 0:H.docs)==null?void 0:N.description}}};var V,A,E,R,T;i.parameters={...i.parameters,docs:{...(V=i.parameters)==null?void 0:V.docs,source:{originalSource:`{
  args: {
    confidence: 0.8
  }
}`,...(E=(A=i.parameters)==null?void 0:A.docs)==null?void 0:E.source},description:{story:"Boundary value at 0.8 (should be medium)",...(T=(R=i.parameters)==null?void 0:R.docs)==null?void 0:T.description}}};var O,P,W,_,G;a.parameters={...a.parameters,docs:{...(O=a.parameters)==null?void 0:O.docs,source:{originalSource:`{
  args: {
    confidence: 0.5
  }
}`,...(W=(P=a.parameters)==null?void 0:P.docs)==null?void 0:W.source},description:{story:"Boundary value at 0.5 (should be medium)",...(G=(_=a.parameters)==null?void 0:_.docs)==null?void 0:G.description}}};var K,Y,q,z,D;d.parameters={...d.parameters,docs:{...(K=d.parameters)==null?void 0:K.docs,source:{originalSource:`{
  args: {
    confidence: 1.0
  }
}`,...(q=(Y=d.parameters)==null?void 0:Y.docs)==null?void 0:q.source},description:{story:"Very high confidence (100%)",...(D=(z=d.parameters)==null?void 0:z.docs)==null?void 0:D.description}}};var F,J,Q,U,X;t.parameters={...t.parameters,docs:{...(F=t.parameters)==null?void 0:F.docs,source:{originalSource:`{
  args: {
    confidence: 0.05
  }
}`,...(Q=(J=t.parameters)==null?void 0:J.docs)==null?void 0:Q.source},description:{story:"Very low confidence (near 0%)",...(X=(U=t.parameters)==null?void 0:U.docs)==null?void 0:X.description}}};var Z,$,ee,ne,oe;p.parameters={...p.parameters,docs:{...(Z=p.parameters)==null?void 0:Z.docs,source:{originalSource:`{
  args: {
    confidence: 0.75
  },
  render: () => <div className="flex flex-col gap-4">
      <ConfidenceIndicator confidence={0.95} />
      <ConfidenceIndicator confidence={0.85} />
      <ConfidenceIndicator confidence={0.75} />
      <ConfidenceIndicator confidence={0.65} />
      <ConfidenceIndicator confidence={0.55} />
      <ConfidenceIndicator confidence={0.45} />
      <ConfidenceIndicator confidence={0.35} />
      <ConfidenceIndicator confidence={0.25} />
      <ConfidenceIndicator confidence={0.15} />
    </div>
}`,...(ee=($=p.parameters)==null?void 0:$.docs)==null?void 0:ee.source},description:{story:"Multiple indicators showing different confidence levels",...(oe=(ne=p.parameters)==null?void 0:ne.docs)==null?void 0:oe.description}}};var ce,re,se,ie,ae;l.parameters={...l.parameters,docs:{...(ce=l.parameters)==null?void 0:ce.docs,source:{originalSource:`{
  args: {
    confidence: 0.75,
    className: 'shadow-lg'
  }
}`,...(se=(re=l.parameters)==null?void 0:re.docs)==null?void 0:se.source},description:{story:"Indicators with custom className",...(ae=(ie=l.parameters)==null?void 0:ie.docs)==null?void 0:ae.description}}};const ue=["HighConfidence","MediumConfidence","LowConfidence","Clickable","BoundaryHigh","BoundaryLow","Perfect","VeryLow","MultipleIndicators","WithCustomClass"];export{i as BoundaryHigh,a as BoundaryLow,s as Clickable,o as HighConfidence,r as LowConfidence,c as MediumConfidence,p as MultipleIndicators,d as Perfect,t as VeryLow,l as WithCustomClass,ue as __namedExportsOrder,fe as default};
