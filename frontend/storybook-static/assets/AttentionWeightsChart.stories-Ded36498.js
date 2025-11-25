import{j as t}from"./jsx-runtime-Bywvkw1S.js";import{A as P}from"./AttentionWeightsChart-B5l3Q2E_.js";import{r as G}from"./index-CleY8y_P.js";import"./cn-CytzSlOG.js";import"./useMediaQuery-Dj87e1Qj.js";import"./throttle-BoqV9_75.js";import"./_commonjsHelpers-Cpj98o6Y.js";import"./mapValues-yqX3NJIo.js";import"./tiny-invariant-CopsF_GD.js";import"./isPlainObject-CjvgKE_V.js";import"./_baseUniq-DznNr2XC.js";const ee={title:"Components/AttentionWeightsChart",component:P,parameters:{layout:"padded",docs:{description:{component:`AttentionWeightsChart visualizes model attention weights for user and gift features.

## Chart Types
- **Bar Chart**: Best for comparing individual feature weights
- **Radar Chart**: Best for seeing overall distribution pattern

## Features
- Toggle between bar and radar chart
- Weights displayed as percentages
- Hover tooltips with full values
- Separate charts for user and gift features
- Responsive layout

## User Features
- Hobbies
- Budget
- Age
- Occasion

## Gift Features
- Category
- Price
- Rating

## Accessibility
- ARIA labels for charts
- Keyboard navigable toggle
- Alt text for chart elements

## Usage
Used in ReasoningPanel to show attention weights reasoning.`}}},tags:["autodocs"],argTypes:{attentionWeights:{description:"Attention weights for user and gift features"},chartType:{description:"Chart visualization type",control:{type:"radio"},options:["bar","radar"]},onChartTypeChange:{description:"Callback when chart type is changed",action:"chart type changed"}}},r=e=>{const[k,q]=G.useState(e.chartType||"bar");return t.jsx(P,{...e,chartType:k,onChartTypeChange:q})},a={render:e=>t.jsx(r,{...e}),args:{attentionWeights:{user_features:{hobbies:.4,budget:.3,age:.2,occasion:.1},gift_features:{category:.5,price:.3,rating:.2}},chartType:"bar"}},n={render:e=>t.jsx(r,{...e}),args:{attentionWeights:{user_features:{hobbies:.35,budget:.25,age:.25,occasion:.15},gift_features:{category:.45,price:.35,rating:.2}},chartType:"bar"}},s={render:e=>t.jsx(r,{...e}),args:{attentionWeights:{user_features:{hobbies:.35,budget:.25,age:.25,occasion:.15},gift_features:{category:.45,price:.35,rating:.2}},chartType:"radar"}},i={render:e=>t.jsx(r,{...e}),args:{attentionWeights:{user_features:{hobbies:.7,budget:.15,age:.1,occasion:.05},gift_features:{category:.6,price:.25,rating:.15}},chartType:"bar"}},o={render:e=>t.jsx(r,{...e}),args:{attentionWeights:{user_features:{hobbies:.25,budget:.25,age:.25,occasion:.25},gift_features:{category:.33,price:.33,rating:.34}},chartType:"bar"}},c={render:e=>t.jsx(r,{...e}),args:{attentionWeights:{user_features:{hobbies:.05,budget:.05,age:.05,occasion:.85},gift_features:{category:.1,price:.1,rating:.8}},chartType:"radar"}};var g,h,p,d,u;a.parameters={...a.parameters,docs:{...(g=a.parameters)==null?void 0:g.docs,source:{originalSource:`{
  render: args => <AttentionWeightsChartWithState {...args} />,
  args: {
    attentionWeights: {
      user_features: {
        hobbies: 0.4,
        budget: 0.3,
        age: 0.2,
        occasion: 0.1
      },
      gift_features: {
        category: 0.5,
        price: 0.3,
        rating: 0.2
      }
    },
    chartType: 'bar'
  }
}`,...(p=(h=a.parameters)==null?void 0:h.docs)==null?void 0:p.source},description:{story:`Default attention weights with bar chart
Shows typical weight distribution`,...(u=(d=a.parameters)==null?void 0:d.docs)==null?void 0:u.description}}};var f,b,l,m,y;n.parameters={...n.parameters,docs:{...(f=n.parameters)==null?void 0:f.docs,source:{originalSource:`{
  render: args => <AttentionWeightsChartWithState {...args} />,
  args: {
    attentionWeights: {
      user_features: {
        hobbies: 0.35,
        budget: 0.25,
        age: 0.25,
        occasion: 0.15
      },
      gift_features: {
        category: 0.45,
        price: 0.35,
        rating: 0.2
      }
    },
    chartType: 'bar'
  }
}`,...(l=(b=n.parameters)==null?void 0:b.docs)==null?void 0:l.source},description:{story:`Bar chart visualization
Best for comparing individual feature weights`,...(y=(m=n.parameters)==null?void 0:m.docs)==null?void 0:y.description}}};var W,C,_,T,A;s.parameters={...s.parameters,docs:{...(W=s.parameters)==null?void 0:W.docs,source:{originalSource:`{
  render: args => <AttentionWeightsChartWithState {...args} />,
  args: {
    attentionWeights: {
      user_features: {
        hobbies: 0.35,
        budget: 0.25,
        age: 0.25,
        occasion: 0.15
      },
      gift_features: {
        category: 0.45,
        price: 0.35,
        rating: 0.2
      }
    },
    chartType: 'radar'
  }
}`,...(_=(C=s.parameters)==null?void 0:C.docs)==null?void 0:_.source},description:{story:`Radar chart visualization
Best for seeing overall distribution pattern`,...(A=(T=s.parameters)==null?void 0:T.docs)==null?void 0:A.description}}};var w,S,v,x,B;i.parameters={...i.parameters,docs:{...(w=i.parameters)==null?void 0:w.docs,source:{originalSource:`{
  render: args => <AttentionWeightsChartWithState {...args} />,
  args: {
    attentionWeights: {
      user_features: {
        hobbies: 0.7,
        budget: 0.15,
        age: 0.1,
        occasion: 0.05
      },
      gift_features: {
        category: 0.6,
        price: 0.25,
        rating: 0.15
      }
    },
    chartType: 'bar'
  }
}`,...(v=(S=i.parameters)==null?void 0:S.docs)==null?void 0:v.source},description:{story:`High hobby weight scenario
Model heavily prioritizes user hobbies`,...(B=(x=i.parameters)==null?void 0:x.docs)==null?void 0:B.description}}};var j,R,H,M,z;o.parameters={...o.parameters,docs:{...(j=o.parameters)==null?void 0:j.docs,source:{originalSource:`{
  render: args => <AttentionWeightsChartWithState {...args} />,
  args: {
    attentionWeights: {
      user_features: {
        hobbies: 0.25,
        budget: 0.25,
        age: 0.25,
        occasion: 0.25
      },
      gift_features: {
        category: 0.33,
        price: 0.33,
        rating: 0.34
      }
    },
    chartType: 'bar'
  }
}`,...(H=(R=o.parameters)==null?void 0:R.docs)==null?void 0:H.source},description:{story:`Balanced weights across all features
Model considers all features equally`,...(z=(M=o.parameters)==null?void 0:M.docs)==null?void 0:z.description}}};var D,E,F,U,O;c.parameters={...c.parameters,docs:{...(D=c.parameters)==null?void 0:D.docs,source:{originalSource:`{
  render: args => <AttentionWeightsChartWithState {...args} />,
  args: {
    attentionWeights: {
      user_features: {
        hobbies: 0.05,
        budget: 0.05,
        age: 0.05,
        occasion: 0.85
      },
      gift_features: {
        category: 0.1,
        price: 0.1,
        rating: 0.8
      }
    },
    chartType: 'radar'
  }
}`,...(F=(E=c.parameters)==null?void 0:E.docs)==null?void 0:F.source},description:{story:`Minimal weights on most features
Model heavily focuses on specific features (occasion, rating)`,...(O=(U=c.parameters)==null?void 0:U.docs)==null?void 0:O.description}}};const te=["Default","BarChart","RadarChart","HighHobbyWeight","BalancedWeights","MinimalWeights"];export{o as BalancedWeights,n as BarChart,a as Default,i as HighHobbyWeight,c as MinimalWeights,s as RadarChart,te as __namedExportsOrder,ee as default};
